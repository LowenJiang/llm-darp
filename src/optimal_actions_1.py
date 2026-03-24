"""
Optimal DARP perturbation actions via joint Gurobi MIP.

Given full knowledge of user flexibility preferences (from CSV), finds
the best time-window perturbation per request to minimise total routing cost.

Approach:
  1. Generate requests using SFGenerator
  2. Look up accepted actions per request from traveler_decisions_augmented.csv
  3. Solve baseline routing (no perturbation) with Gurobi MIP
  4. Solve a single joint MIP that simultaneously optimizes routing (arc
     variables x[i,j,k], visit-time B[i,k]) and action selection (binary
     y[r,a]).  Decision-dependent time windows are linear in y, so the
     full problem remains a linear MIP.
  5. Report per-request optimal actions and total cost savings

Usage:
    python src/optimal_actions.py [--customers 30] [--vehicles 5] [--time-limit 10] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from oracle_generator import SFGenerator  # noqa: E402
import gurobipy as gp                     # noqa: E402
from gurobipy import GRB                  # noqa: E402

# ---------------------------------------------------------------------------
# Constants  (must match meta_train.py / dvrp_env.py)
# ---------------------------------------------------------------------------
ACTION_SPACE_MAP: list[tuple[int, int]] = [
    (-30,  0), (-30, 10), (-30, 20), (-30, 30),
    (-20,  0), (-20, 10), (-20, 20), (-20, 30),
    (-10,  0), (-10, 10), (-10, 20), (-10, 30),
    (  0,  0), (  0, 10), (  0, 20), (  0, 30),
]

FLEX_COLUMNS: list[str] = [
    "flexible for both early pickup and late dropoff",
    "flexible for early pickup, but inflexible for late dropoff",
    "flexible for late dropoff, but inflexible for early pickup",
    "inflexible for any schedule changes",
]

FLEX_STR_TO_IDX: dict[str, int] = {
    "flexible for both early pickup and late dropoff": 0,
    "flexible for early pickup, but inflexible for late dropoff": 1,
    "flexible for late dropoff, but inflexible for early pickup": 2,
    "inflexible for any schedule changes": 3,
}

# Penalty for leaving a request unserved (minutes).  Set high enough that
# the solver will always serve a request when physically feasible.
UNSERVED_PENALTY: float = 100_000.0


# ===================================================================
# 1.  Data helpers
# ===================================================================

def build_decision_lookup(csv_path: Path) -> dict:
    """Build a fast lookup from *traveler_decisions_augmented.csv*."""
    df = pd.read_csv(csv_path)
    lookup: dict[tuple, dict[str, bool]] = {}
    for _, row in df.iterrows():
        key = (
            int(row["traveler_id"]),
            str(row["trip_purpose"]).strip(),
            str(row["departure_location"]).strip(),
            str(row["arrival_location"]).strip(),
            int(row["pickup_shift_min"]),
            int(row["dropoff_shift_min"]),
        )
        decisions = {
            col: str(row[col]).strip().lower() == "accept"
            for col in FLEX_COLUMNS
        }
        lookup[key] = decisions
    return lookup


def get_accepted_actions(
    traveler_id: int,
    trip_meta: dict,
    decision_lookup: dict,
    flexibility_str: str,
) -> list[tuple[int, int, int]]:
    """Return accepted *(action_idx, pickup_shift, dropoff_shift)* tuples."""
    tp = str(trip_meta["trip_purpose"]).strip()
    dep = str(trip_meta["departure_location"]).strip()
    arr = str(trip_meta["arrival_location"]).strip()

    accepted: list[tuple[int, int, int]] = []
    for a_idx, (ps, ds) in enumerate(ACTION_SPACE_MAP):
        key = (traveler_id, tp, dep, arr, abs(ps), abs(ds))
        decisions = decision_lookup.get(key)
        if decisions is not None and decisions.get(flexibility_str, False):
            accepted.append((a_idx, ps, ds))

    # Guarantee (0, 0) is always available
    if not any(a[0] == 12 for a in accepted):
        accepted.append((12, 0, 0))
    return accepted


def _prepare_td(td):
    """Ensure the TensorDict has the keys the solver expects."""
    if "vehicle_capacity" not in td.keys() and "capacity" in td.keys():
        cap = td["capacity"]
        td.set("vehicle_capacity",
               cap.unsqueeze(-1) if cap.dim() == 1 else cap)
    return td


def _apply_actions(td, action_dict: dict[int, int]):
    """Return a copy of *td* with perturbed time windows."""
    td2 = td.clone()
    tw = td2["time_windows"].clone()
    for r, a_idx in action_dict.items():
        ps, ds = ACTION_SPACE_MAP[a_idx]
        p, d = 2 * r - 1, 2 * r
        tw[0, p, 0] += ps
        tw[0, p, 1] += ps
        tw[0, d, 0] += ds
        tw[0, d, 1] += ds
    tw = torch.clamp(tw, min=30, max=1410)
    td2["time_windows"] = tw
    return _prepare_td(td2)


def _build_arcs_and_bigm(N, dep_end, e, ld, c):
    """Build the feasible arc set, per-arc Big-M values, and adjacency lists.

    Ensures structurally necessary arcs always exist:
      - depot (0) → end-depot for empty vehicle routes
      - every node → end-depot  so vehicles can always return
      - depot (0) → every pickup  so vehicles can always start serving
    """
    arcs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(i: int, j: int):
        if (i, j) not in seen:
            arcs.append((i, j))
            seen.add((i, j))

    # --- structurally required arcs (always included) ---------------------
    _add(0, dep_end)                       # empty vehicle path
    for i in N:
        if i != dep_end and i != 0:
            _add(i, dep_end)               # every node can reach end-depot
            _add(0, i)                     # depot can reach every node

    # --- time-feasible arcs -----------------------------------------------
    for i in N:
        for j in N:
            if i == j:              continue
            if i == dep_end:        continue   # no arcs out of end depot
            if j == 0:              continue   # no arcs into start depot
            if e[i] + c[i, j] > ld[j]:        # time-infeasible arc
                continue
            _add(i, j)

    M = {(i, j): max(0.0, ld[i] + c[i, j] - e[j]) for (i, j) in arcs}

    out_nbrs: dict[int, list[int]] = {i: [] for i in N}
    in_nbrs:  dict[int, list[int]] = {j: [] for j in N}
    for (i, j) in arcs:
        out_nbrs[i].append(j)
        in_nbrs[j].append(i)

    return arcs, M, out_nbrs, in_nbrs


def _solve_routing(td, max_vehicles: int, time_limit: int = 5):
    """Solve the DARP instance in *td* with Gurobi and return ``(cost, routes)``.

    Uses a tightened Cordeau-style formulation with per-arc Big-M values,
    infeasible-arc elimination, and precedence cuts for a strong LP relaxation.

    A soft-service formulation (binary ``z[r]`` with large penalty) guarantees
    a feasible MIP regardless of time-window tightness or vehicle count.
    """
    batch_idx = 0

    # ---- data extraction from TensorDict --------------------------------
    h3  = td["h3_indices"][batch_idx].detach().cpu().numpy().astype(int)
    glb = td["travel_time_matrix"][batch_idx].detach().cpu().numpy()
    tt  = glb[np.ix_(h3, h3)]                                # minutes
    tw  = td["time_windows"][batch_idx].detach().cpu().numpy()

    cap_t = td["vehicle_capacity"][batch_idx]
    veh_cap = (int(cap_t.reshape(-1)[0].item())
               if isinstance(cap_t, torch.Tensor) else int(cap_t))

    num_td = h3.shape[0]          # 2n + 1  (depot + pairs)
    n      = (num_td - 1) // 2    # number of requests

    # ---- Gurobi node sets ------------------------------------------------
    #   0          = start depot   (TD node 0)
    #   1  ... n   = pickups       (TD node 2i-1)
    #   n+1 ... 2n = deliveries    (TD node 2i)
    #   2n+1       = end depot     (TD node 0)
    K       = list(range(max_vehicles))
    P       = list(range(1, n + 1))
    D       = list(range(n + 1, 2 * n + 1))
    dep_end = 2 * n + 1
    N       = [0] + P + D + [dep_end]

    def _g2td(g: int) -> int:
        """Map Gurobi node index to TensorDict node index."""
        if g == 0 or g == dep_end:
            return 0
        return (2 * g - 1) if g <= n else 2 * (g - n)

    # cost / travel-time in Gurobi node space (minutes)
    c = {(i, j): 0.0 if i == j else float(tt[_g2td(i), _g2td(j)])
         for i in N for j in N}

    # time windows
    e  = {g: float(tw[_g2td(g), 0]) for g in N}
    ld = {g: float(tw[_g2td(g), 1]) for g in N}

    # ---- Build feasible arc set & per-arc Big-M --------------------------
    arcs, M, out_nbrs, in_nbrs = _build_arcs_and_bigm(N, dep_end, e, ld, c)

    # Big-M for time-window relaxation when a request is unserved
    TW_M = max(ld[g] for g in N) + 1.0

    # ---- model -----------------------------------------------------------
    mdl = gp.Model("DARP")
    mdl.setParam("OutputFlag", 0)
    mdl.setParam("TimeLimit", time_limit)

    x: dict[tuple, gp.Var] = {}
    for (i, j) in arcs:
        for k in K:
            x[i, j, k] = mdl.addVar(vtype=GRB.BINARY,
                                      name=f"x_{i}_{j}_{k}")
    B = mdl.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0, name="B")

    # service indicator: z[r] = 1 iff request r is served
    z = mdl.addVars(P, vtype=GRB.BINARY, name="z")

    # objective: minimise total travel time + penalty for unserved
    mdl.setObjective(
        gp.quicksum(c[i, j] * x[i, j, k]
                    for (i, j) in arcs for k in K)
        + UNSERVED_PENALTY * gp.quicksum(1 - z[r] for r in P),
        GRB.MINIMIZE,
    )

    # (1) each pickup visited exactly once *if served*
    for i in P:
        mdl.addConstr(
            gp.quicksum(x[i, j, k]
                        for j in out_nbrs[i] for k in K) == z[i])

    # (2) same vehicle serves pickup i and its delivery n+i
    for i in P:
        for k in K:
            mdl.addConstr(
                gp.quicksum(x[i, j, k] for j in out_nbrs[i])
                - gp.quicksum(x[n + i, j, k]
                              for j in out_nbrs[n + i]) == 0)

    # (3) every vehicle leaves start depot / enters end depot
    for k in K:
        mdl.addConstr(
            gp.quicksum(x[0, j, k] for j in out_nbrs[0]) == 1)
        mdl.addConstr(
            gp.quicksum(x[i, dep_end, k]
                        for i in in_nbrs[dep_end]) == 1)

    # (4) flow conservation at pickup / delivery nodes
    for i in P + D:
        for k in K:
            mdl.addConstr(
                gp.quicksum(x[j, i, k] for j in in_nbrs[i])
                - gp.quicksum(x[i, j, k] for j in out_nbrs[i]) == 0)

    # time-window constraints (relaxed for unserved requests via Big-M)
    for i in P:
        d_node = n + i
        for k in K:
            # pickup
            mdl.addConstr(B[i, k] >= e[i] - TW_M * (1 - z[i]))
            mdl.addConstr(B[i, k] <= ld[i] + TW_M * (1 - z[i]))
            # delivery
            mdl.addConstr(B[d_node, k] >= e[d_node] - TW_M * (1 - z[i]))
            mdl.addConstr(B[d_node, k] <= ld[d_node] + TW_M * (1 - z[i]))

    # depot time windows (always enforced)
    for k in K:
        mdl.addConstr(B[0, k] >= e[0])
        mdl.addConstr(B[0, k] <= ld[0])
        mdl.addConstr(B[dep_end, k] >= e[dep_end])
        mdl.addConstr(B[dep_end, k] <= ld[dep_end])

    # travel-time linking (per-arc Big-M)
    for (i, j) in arcs:
        for k in K:
            mdl.addConstr(
                B[j, k] >= B[i, k] + c[i, j]
                - M[i, j] * (1 - x[i, j, k]))

    # precedence: delivery after pickup + direct travel (valid inequality)
    for r in P:
        d_node = n + r
        t_pd = c[r, d_node]
        M_prec = max(0.0, ld[r] + t_pd - e[d_node])
        for k in K:
            mdl.addConstr(
                B[d_node, k] >= B[r, k] + t_pd
                - M_prec * (1 - gp.quicksum(
                    x[r, j, k] for j in out_nbrs[r])))

    # ---- solve -----------------------------------------------------------
    mdl.optimize()

    if mdl.SolCount > 0:
        # report any unserved requests
        unserved = [r for r in P if z[r].X < 0.5]
        if unserved:
            print(f"  WARNING: {len(unserved)} request(s) unserved: {unserved}")

        routes: list[list[int]] = []
        for k in K:
            route: list[int] = []
            cur = 0
            while True:
                nxt = None
                for j in out_nbrs.get(cur, []):
                    if (cur, j, k) in x and x[cur, j, k].X > 0.5:
                        nxt = j
                        break
                if nxt is None or nxt == dep_end:
                    break
                route.append(_g2td(nxt))
                cur = nxt
            if route:
                routes.append(route)

        # Return cost excluding penalty so it is comparable across runs.
        # If there are unserved requests the raw ObjVal includes the penalty;
        # we separate the two for clarity.
        travel_cost = sum(
            c[i, j] * x[i, j, k].X
            for (i, j) in arcs for k in K
            if x[i, j, k].X > 0.5
        )
        return travel_cost, routes, unserved

    return float("inf"), [], list(P)


# ===================================================================
# 2.  Joint MIP: routing + action selection in one model
# ===================================================================

def solve_joint_mip(
    td,
    n_customers: int,
    accepted_actions: dict[int, list[tuple[int, int, int]]],
    max_vehicles: int = 5,
    time_limit: int = 60,
) -> dict:
    """Solve DARP routing and action selection jointly in a single MIP.

    Uses the same tightened formulation as ``_solve_routing`` (per-arc Big-M,
    arc elimination, precedence cuts) plus binary action-selection variables
    ``y[r, a]`` with decision-dependent time-window constraints.

    A soft-service formulation (binary ``z[r]`` with large penalty) guarantees
    a feasible MIP regardless of time-window tightness or vehicle count.

    Returns dict with keys: total_cost, selected_actions, routes, unserved.
    """
    batch_idx = 0

    # ---- data extraction from TensorDict --------------------------------
    h3  = td["h3_indices"][batch_idx].detach().cpu().numpy().astype(int)
    glb = td["travel_time_matrix"][batch_idx].detach().cpu().numpy()
    tt  = glb[np.ix_(h3, h3)]
    tw  = td["time_windows"][batch_idx].detach().cpu().numpy()

    cap_t = td["vehicle_capacity"][batch_idx]
    veh_cap = (int(cap_t.reshape(-1)[0].item())
               if isinstance(cap_t, torch.Tensor) else int(cap_t))

    num_td = h3.shape[0]
    n      = (num_td - 1) // 2

    # ---- Gurobi node sets ------------------------------------------------
    K       = list(range(max_vehicles))
    P       = list(range(1, n + 1))
    D       = list(range(n + 1, 2 * n + 1))
    dep_end = 2 * n + 1
    N       = [0] + P + D + [dep_end]

    def _g2td(g: int) -> int:
        if g == 0 or g == dep_end:
            return 0
        return (2 * g - 1) if g <= n else 2 * (g - n)

    c = {(i, j): 0.0 if i == j else float(tt[_g2td(i), _g2td(j)])
         for i in N for j in N}

    # base time windows (before any action perturbation)
    e_base  = {g: float(tw[_g2td(g), 0]) for g in N}
    l_base  = {g: float(tw[_g2td(g), 1]) for g in N}

    # ---- Pre-compute clamped TW bounds per (request, action) -------------
    TW_LO = 30.0
    TW_HI = 1410.0

    def _clamp(v):
        return max(TW_LO, min(TW_HI, v))

    # For each request r (1..n), store list of (a_idx, e_p, l_p, e_d, l_d)
    action_tw: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for r in P:
        atw = []
        for a_idx, ps, ds in accepted_actions.get(r, [(12, 0, 0)]):
            ep = _clamp(e_base[r] + ps)
            lp = _clamp(l_base[r] + ps)
            ed = _clamp(e_base[n + r] + ds)
            ld_val = _clamp(l_base[n + r] + ds)
            atw.append((a_idx, ep, lp, ed, ld_val))
        action_tw[r] = atw

    # ---- Widest possible TW per node (over all actions) for arc pruning --
    e_wide: dict[int, float] = dict(e_base)
    l_wide: dict[int, float] = dict(l_base)
    for r in P:
        atw = action_tw[r]
        e_wide[r]     = min(t[1] for t in atw)   # min e_pickup
        l_wide[r]     = max(t[2] for t in atw)   # max l_pickup
        e_wide[n + r] = min(t[3] for t in atw)   # min e_delivery
        l_wide[n + r] = max(t[4] for t in atw)   # max l_delivery

    # ---- Build feasible arc set & per-arc Big-M --------------------------
    arcs, M_arc, out_nbrs, in_nbrs = _build_arcs_and_bigm(
        N, dep_end, e_wide, l_wide, c
    )

    # Big-M for time-window relaxation when a request is unserved
    TW_M = TW_HI + 1.0

    # ---- model -----------------------------------------------------------
    mdl = gp.Model("DARP_Joint")
    mdl.setParam("OutputFlag", 0)
    mdl.setParam("TimeLimit", time_limit)

    x: dict[tuple, gp.Var] = {}
    for (i, j) in arcs:
        for k in K:
            x[i, j, k] = mdl.addVar(vtype=GRB.BINARY,
                                      name=f"x_{i}_{j}_{k}")
    B = mdl.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0, name="B")

    # action-selection variables: y[r, local_index]
    y: dict[tuple, gp.Var] = {}
    for r in P:
        for loc_idx in range(len(action_tw[r])):
            y[r, loc_idx] = mdl.addVar(vtype=GRB.BINARY,
                                        name=f"y_{r}_{loc_idx}")

    # service indicator: z[r] = 1 iff request r is served
    z = mdl.addVars(P, vtype=GRB.BINARY, name="z")

    # objective: minimise total travel time + penalty for unserved
    mdl.setObjective(
        gp.quicksum(c[i, j] * x[i, j, k]
                    for (i, j) in arcs for k in K)
        + UNSERVED_PENALTY * gp.quicksum(1 - z[r] for r in P),
        GRB.MINIMIZE,
    )

    # --- routing constraints ----------------------------------------------

    # (1) each pickup visited exactly once *if served*
    for i in P:
        mdl.addConstr(
            gp.quicksum(x[i, j, k]
                        for j in out_nbrs[i] for k in K) == z[i])

    # (2) same vehicle serves pickup i and delivery n+i
    for i in P:
        for k in K:
            mdl.addConstr(
                gp.quicksum(x[i, j, k] for j in out_nbrs[i])
                - gp.quicksum(x[n + i, j, k]
                              for j in out_nbrs[n + i]) == 0)

    # (3) every vehicle leaves start depot / enters end depot
    for k in K:
        mdl.addConstr(
            gp.quicksum(x[0, j, k] for j in out_nbrs[0]) == 1)
        mdl.addConstr(
            gp.quicksum(x[i, dep_end, k]
                        for i in in_nbrs[dep_end]) == 1)

    # (4) flow conservation at pickup / delivery nodes
    for i in P + D:
        for k in K:
            mdl.addConstr(
                gp.quicksum(x[j, i, k] for j in in_nbrs[i])
                - gp.quicksum(x[i, j, k] for j in out_nbrs[i]) == 0)

    # travel-time linking (per-arc Big-M)
    for (i, j) in arcs:
        for k in K:
            mdl.addConstr(
                B[j, k] >= B[i, k] + c[i, j]
                - M_arc[i, j] * (1 - x[i, j, k]))

    # depot time windows (no action dependence, always enforced)
    for k in K:
        mdl.addConstr(B[0, k] >= e_base[0])
        mdl.addConstr(B[0, k] <= l_base[0])
        mdl.addConstr(B[dep_end, k] >= e_base[dep_end])
        mdl.addConstr(B[dep_end, k] <= l_base[dep_end])

    # --- action-selection constraints -------------------------------------

    # exactly one action per request *if served*; no action if unserved
    for r in P:
        mdl.addConstr(
            gp.quicksum(y[r, li] for li in range(len(action_tw[r]))) == z[r])

    # decision-dependent time windows for pickups
    #   When z[r]=1: exactly one y active → B bounded by that action's TW
    #   When z[r]=0: all y=0, sum=0 → Big-M relaxes the constraints
    for r in P:
        atw = action_tw[r]
        for k in K:
            mdl.addConstr(
                B[r, k] >= gp.quicksum(
                    atw[li][1] * y[r, li] for li in range(len(atw)))
                - TW_M * (1 - z[r]))
            mdl.addConstr(
                B[r, k] <= gp.quicksum(
                    atw[li][2] * y[r, li] for li in range(len(atw)))
                + TW_M * (1 - z[r]))

    # decision-dependent time windows for deliveries
    for r in P:
        atw = action_tw[r]
        d_node = n + r
        for k in K:
            mdl.addConstr(
                B[d_node, k] >= gp.quicksum(
                    atw[li][3] * y[r, li] for li in range(len(atw)))
                - TW_M * (1 - z[r]))
            mdl.addConstr(
                B[d_node, k] <= gp.quicksum(
                    atw[li][4] * y[r, li] for li in range(len(atw)))
                + TW_M * (1 - z[r]))

    # precedence: delivery after pickup + direct travel (valid inequality)
    for r in P:
        d_node = n + r
        t_pd = c[r, d_node]
        M_prec = max(0.0, l_wide[r] + t_pd - e_wide[d_node])
        for k in K:
            mdl.addConstr(
                B[d_node, k] >= B[r, k] + t_pd
                - M_prec * (1 - gp.quicksum(
                    x[r, j, k] for j in out_nbrs[r])))

    # ---- solve -----------------------------------------------------------
    mdl.optimize()

    if mdl.SolCount > 0:
        # extract service indicators
        unserved = [r for r in P if z[r].X < 0.5]
        if unserved:
            print(f"  WARNING: {len(unserved)} request(s) unserved: {unserved}")

        # extract selected actions
        selected: dict[int, int] = {}
        for r in P:
            if z[r].X < 0.5:
                selected[r] = 12  # unserved → default action (irrelevant)
                continue
            atw = action_tw[r]
            for li in range(len(atw)):
                if y[r, li].X > 0.5:
                    selected[r] = atw[li][0]  # a_idx
                    break
            else:
                selected[r] = 12  # fallback

        # extract routes
        routes: list[list[int]] = []
        for k in K:
            route: list[int] = []
            cur = 0
            while True:
                nxt = None
                for j in out_nbrs.get(cur, []):
                    if (cur, j, k) in x and x[cur, j, k].X > 0.5:
                        nxt = j
                        break
                if nxt is None or nxt == dep_end:
                    break
                route.append(_g2td(nxt))
                cur = nxt
            if route:
                routes.append(route)

        # Compute travel cost excluding the unserved penalty
        travel_cost = sum(
            c[i, j] * x[i, j, k].X
            for (i, j) in arcs for k in K
            if x[i, j, k].X > 0.5
        )

        return {
            "total_cost": travel_cost,
            "selected_actions": selected,
            "routes": routes,
            "unserved": unserved,
        }

    return {
        "total_cost": float("inf"),
        "selected_actions": {r: 12 for r in P},
        "routes": [],
        "unserved": list(P),
    }


# ===================================================================
# 3.  Coordinate descent solver (retained for baseline / fix_action)
# ===================================================================

def solve_optimal(
    td,
    n_customers: int,
    accepted_actions: dict[int, list[tuple[int, int, int]]],
    max_vehicles: int = 5,
    time_limit_per_solve: int = 5,
    max_iterations: int = 10,
    fix_action: int | None = None,
) -> dict:
    """Find best perturbation actions via greedy coordinate descent.

    If *fix_action* is set, forces that action for every request (used
    for baseline evaluation).
    """
    if fix_action is not None:
        actions = {r: fix_action for r in range(1, n_customers + 1)}
        td_mod = _apply_actions(td, actions)
        cost, routes, unserved = _solve_routing(
            td_mod, max_vehicles,
            time_limit=max(time_limit_per_solve, 10),
        )
        return {
            "total_cost": cost,
            "selected_actions": actions,
            "routes": routes,
            "unserved": unserved,
        }

    # --- Start with no perturbation ---
    current = {r: 12 for r in range(1, n_customers + 1)}
    td_cur = _apply_actions(td, current)
    best_cost, best_routes, best_unserved = _solve_routing(
        td_cur, max_vehicles, time_limit=time_limit_per_solve,
    )
    print(f"  Initial cost (no perturbation): {best_cost:.2f} min")
    if best_unserved:
        print(f"  Initial unserved: {best_unserved}")

    for it in range(1, max_iterations + 1):
        improved = False
        for r in range(1, n_customers + 1):
            cur_a = current[r]
            for a_idx, ps, ds in accepted_actions.get(r, [(12, 0, 0)]):
                if a_idx == cur_a:
                    continue
                trial = current.copy()
                trial[r] = a_idx
                td_trial = _apply_actions(td, trial)
                cost, routes, unserved = _solve_routing(
                    td_trial, max_vehicles,
                    time_limit=time_limit_per_solve,
                )
                if cost < best_cost - 0.01:
                    best_cost = cost
                    best_routes = routes
                    best_unserved = unserved
                    current[r] = a_idx
                    improved = True
                    print(f"  Iter {it}: Req {r} → act {a_idx} "
                          f"({ps:+d},{ds:+d}), cost={cost:.2f}")
        if not improved:
            print(f"  Converged after {it} iteration(s).")
            break

    return {
        "total_cost": best_cost,
        "selected_actions": current,
        "routes": best_routes,
        "unserved": best_unserved,
    }


# ===================================================================
# 4.  Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimal DARP perturbation via Gurobi MIP search"
    )
    parser.add_argument("--customers", type=int, default=30)
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=60,
                        help="Gurobi time limit per routing solve (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_cust = args.customers
    csv_path = SRC_DIR / "traveler_trip_types_res_7.csv"
    ttm_path = SRC_DIR / "travel_time_matrix_res_7.csv"
    decisions_csv = SRC_DIR / "traveler_decisions_augmented.csv"

    # --- 1. Generate instance ---
    print(f"Generating {n_cust}-customer instance (seed={args.seed}) …")
    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        num_customers=n_cust,
        perturbation=0,
        seed=args.seed,
    )
    td = generator(batch_size=[1])
    td = _prepare_td(td)

    tw = td["time_windows"][0].cpu().numpy()
    user_ids = td["user_id"][0].cpu().numpy()
    N = tw.shape[0]
    print(f"  Nodes = {N}  (1 depot + {n_cust} pairs)")

    # --- 2. Build accepted actions ---
    print("Loading traveler decisions …")
    decision_lookup = build_decision_lookup(decisions_csv)

    trip_metadata = td["trip_metadata"][0]
    flex_raw = td["flexibility"][0].cpu().numpy()
    idx_to_flex = {v: k for k, v in FLEX_STR_TO_IDX.items()}

    accepted_per_request: dict[int, list[tuple[int, int, int]]] = {}
    for r in range(1, n_cust + 1):
        p_node = 2 * r - 1
        traveler_id = int(user_ids[p_node])
        flex_idx = int(flex_raw[p_node])
        flex_str = idx_to_flex.get(flex_idx, "inflexible for any schedule changes")
        meta = trip_metadata.get(traveler_id)
        if meta is None:
            accepted_per_request[r] = [(12, 0, 0)]
            continue
        accepted_per_request[r] = get_accepted_actions(
            traveler_id, meta, decision_lookup, flex_str)

    total_acc = sum(len(v) for v in accepted_per_request.values())
    print(f"  Accepted actions: min={min(len(v) for v in accepted_per_request.values())}  "
          f"max={max(len(v) for v in accepted_per_request.values())}  total={total_acc}")

    # --- 3. Baseline ---
    print("\n" + "=" * 60)
    print("BASELINE  (action 12 = no perturbation)")
    print("=" * 60)
    baseline = solve_optimal(
        td, n_cust, accepted_per_request,
        max_vehicles=args.vehicles,
        time_limit_per_solve=args.time_limit,
        fix_action=12,
    )
    print(f"\nBaseline cost : {baseline['total_cost']:.2f} min")
    print(f"  Vehicles    : {len(baseline['routes'])}")
    if baseline["unserved"]:
        print(f"  UNSERVED    : {baseline['unserved']}")
    for v, route in enumerate(baseline["routes"]):
        print(f"  Vehicle {v + 1}  : [0] → {route} → [0]")

    # --- 4. Optimal (joint MIP) ---
    print("\n" + "=" * 60)
    print("OPTIMAL  (joint MIP: routing + action selection)")
    print("=" * 60)
    optimal = solve_joint_mip(
        td, n_cust, accepted_per_request,
        max_vehicles=args.vehicles,
        time_limit=args.time_limit,
    )
    print(f"\nOptimal cost  : {optimal['total_cost']:.2f} min")
    print(f"  Vehicles    : {len(optimal['routes'])}")
    if optimal["unserved"]:
        print(f"  UNSERVED    : {optimal['unserved']}")
    for v, route in enumerate(optimal["routes"]):
        print(f"  Vehicle {v + 1}  : [0] → {route} → [0]")

    # --- 5. Report ---
    print("\n" + "=" * 60)
    print("PER-REQUEST OPTIMAL ACTIONS")
    print("=" * 60)
    print(f"{'Req':>4} {'Trav':>5} {'Srv':>4} {'Act':>4} {'PShift':>7} {'DShift':>7}  "
          f"{'Pickup TW':>20}  {'Delivery TW':>20}")
    print("-" * 88)
    for r in range(1, n_cust + 1):
        p, d = 2 * r - 1, 2 * r
        a_idx = optimal["selected_actions"].get(r, 12)
        ps, ds = ACTION_SPACE_MAP[a_idx]
        trav_id = int(user_ids[p])
        served = "Y" if r not in optimal.get("unserved", []) else "N"
        print(
            f"{r:4d} {trav_id:5d} {served:>4s} {a_idx:4d} {ps:+5d}m  {ds:+5d}m   "
            f"[{tw[p,0]+ps:6.0f}, {tw[p,1]+ps:6.0f}]        "
            f"[{tw[d,0]+ds:6.0f}, {tw[d,1]+ds:6.0f}]"
        )

    savings = baseline["total_cost"] - optimal["total_cost"]
    pct = 100 * savings / baseline["total_cost"] if baseline["total_cost"] > 0 else 0
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline cost (V_0*) : {baseline['total_cost']:>10.2f} min")
    print(f"  Optimal  cost (V*)   : {optimal['total_cost']:>10.2f} min")
    print(f"  Savings  (V_0*-V*)   : {savings:>10.2f} min  ({pct:.2f}%)")
    print(f"  Baseline vehicles    : {len(baseline['routes'])}")
    print(f"  Optimal  vehicles    : {len(optimal['routes'])}")
    print(f"  Baseline unserved    : {len(baseline.get('unserved', []))}")
    print(f"  Optimal  unserved    : {len(optimal.get('unserved', []))}")


if __name__ == "__main__":
    main()