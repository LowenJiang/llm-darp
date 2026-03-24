"""OR-Tools DARP perturbation solver using relaxation + rounding.

Approach (fast heuristic alternative to the joint Gurobi MIP in optimal_actions.py):
  1. Solve baseline DARP (no perturbation) with OR-Tools
  2. For each request, widen time windows by customer flexibility:
     - Pickup earliest shifted earlier by max accepted pickup slack
     - Dropoff latest shifted later by max accepted dropoff slack
     - Pickup latest and dropoff earliest stay unchanged
  3. Re-solve the slacked (widened) DARP — this is a lower bound
  4. Extract visit times from the slacked solution
  5. Retroactively select the smallest discrete action that keeps each
     request's visit time feasible within the shifted window
  6. Re-solve with the actual discrete actions for the true cost
  7. Report baseline vs perturbed cost

Usage:
    python src/or_tools_pert.py [--customers 30] [--vehicles 5] [--time-limit 30] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from oracle_generator import SFGenerator  # noqa: E402
from optimal_actions import (  # noqa: E402
    FLEX_STR_TO_IDX,
    build_decision_lookup,
    get_accepted_actions,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_SCALE = 100  # float-minutes → int for OR-Tools


# ===================================================================
# 1.  Helpers
# ===================================================================

def _prepare_td(td):
    """Ensure the TensorDict has ``vehicle_capacity`` key."""
    if "vehicle_capacity" not in td.keys() and "capacity" in td.keys():
        cap = td["capacity"]
        td.set("vehicle_capacity",
               cap.unsqueeze(-1) if cap.dim() == 1 else cap)
    return td


_SHIFT_STEPS_LIST = [-30, -20, -10, 0, 10, 20, 30]


def _apply_actions(td, shift_dict: Dict[int, Tuple[int, int]]):
    """Return a copy of *td* with time windows shifted by raw (ps, ds) pairs."""
    td2 = td.clone()
    tw = td2["time_windows"].clone()
    for r, (ps, ds) in shift_dict.items():
        p, d = 2 * r - 1, 2 * r
        tw[0, p, 0] += ps
        tw[0, p, 1] += ps
        tw[0, d, 0] += ds
        tw[0, d, 1] += ds
    tw = torch.clamp(tw, min=30, max=1410)
    td2["time_windows"] = tw
    return _prepare_td(td2)


# ===================================================================
# 2.  OR-Tools DARP solver (returns visit times)
# ===================================================================

def darp_solve(
    td: Any,
    max_vehicles: int = 5,
    time_limit_seconds: int = 30,
    vehicle_cost: float = 0.0,
) -> Dict[str, Any]:
    """Solve a DARP instance with OR-Tools and return visit times per node.

    Returns dict with keys: total_distance, vehicles_used, routes,
    visit_times (node → minutes), status.
    """
    batch_idx = 0

    # ---- data extraction ---------------------------------------------------
    h3_idx = td["h3_indices"][batch_idx].detach().cpu().numpy().astype(int)
    num_nodes = h3_idx.shape[0]

    glb = td["travel_time_matrix"][batch_idx].detach().cpu().numpy()
    tt_float = glb[np.ix_(h3_idx, h3_idx)]  # minutes
    tt_scaled = np.maximum(np.rint(tt_float * TIME_SCALE).astype(int), 0)

    tw = td["time_windows"][batch_idx].detach().cpu().numpy()
    tw_scaled = (tw * TIME_SCALE).astype(int)

    cap_t = td["vehicle_capacity"][batch_idx]
    veh_cap = (int(cap_t.reshape(-1)[0].item())
               if isinstance(cap_t, torch.Tensor) else int(cap_t))

    raw_demand = td["demand"][batch_idx].detach().cpu().numpy()
    demand_int = np.rint(raw_demand).astype(int)
    num_requests = (num_nodes - 1) // 2

    # ---- OR-Tools model ----------------------------------------------------
    manager = pywrapcp.RoutingIndexManager(num_nodes, max_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Arc cost = travel time
    def _time_cb(from_index: int, to_index: int) -> int:
        return int(tt_scaled[manager.IndexToNode(from_index),
                             manager.IndexToNode(to_index)])

    transit_idx = routing.RegisterTransitCallback(_time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    if vehicle_cost > 0:
        routing.SetFixedCostOfAllVehicles(int(vehicle_cost * TIME_SCALE))

    # Time dimension
    horizon = int(tw_scaled[:, 1].max() + 60 * TIME_SCALE)
    routing.AddDimension(transit_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node in range(num_nodes):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(
            int(tw_scaled[node, 0]), int(tw_scaled[node, 1]))

    # Capacity dimension
    def _demand_cb(from_index: int) -> int:
        return int(demand_int[manager.IndexToNode(from_index)])

    dem_idx = routing.RegisterUnaryTransitCallback(_demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_idx, 0, [veh_cap] * max_vehicles, True, "Capacity")

    # Pickup & delivery constraints
    solver = routing.solver()
    for i in range(num_requests):
        p_idx = manager.NodeToIndex(2 * i + 1)
        d_idx = manager.NodeToIndex(2 * i + 2)
        routing.AddPickupAndDelivery(p_idx, d_idx)
        solver.Add(routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx))
        solver.Add(time_dim.CumulVar(p_idx) <= time_dim.CumulVar(d_idx))

    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = time_limit_seconds
    params.log_search = False

    solution = routing.SolveWithParameters(params)

    # ---- extract results ---------------------------------------------------
    if solution is None:
        return {
            "total_distance": float("inf"),
            "vehicles_used": 0,
            "routes": [],
            "visit_times": {},
            "status": "infeasible",
        }

    routes: List[List[int]] = []
    total_dist = 0.0
    vehicles_used = 0
    visit_times: Dict[int, float] = {}

    for vid in range(max_vehicles):
        index = routing.Start(vid)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue

        route: List[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            visit_times[node] = solution.Value(
                time_dim.CumulVar(index)) / TIME_SCALE

            if node != 0:
                route.append(node)

            prev = index
            index = solution.Value(routing.NextVar(index))
            total_dist += tt_float[manager.IndexToNode(prev),
                                   manager.IndexToNode(index)]

        # end-depot time
        node = manager.IndexToNode(index)
        visit_times.setdefault(
            node, solution.Value(time_dim.CumulVar(index)) / TIME_SCALE)

        if route:
            routes.append(route)
            vehicles_used += 1

    return {
        "total_distance": total_dist,
        "vehicles_used": vehicles_used,
        "routes": routes,
        "visit_times": visit_times,
        "status": "optimal",
    }


# ===================================================================
# 3.  Widen time windows (relaxation step)
# ===================================================================

def widen_time_windows(
    td,
    accepted_per_request: Dict[int, List[Tuple[int, int, int]]],
    n_customers: int,
):
    """Return a copy of *td* with widened time windows.

    Expanded action space: shifts in {-30..+30} (10-min steps).
    User preferences limit early pickup (ps<0) and late dropoff (ds>0).
    Delay pickup (ps>0) and advance dropoff (ds<0) are always permissible.

    Per request the window becomes:
      - pickup:  (ep + min(ps_accepted),  lp + 30)
      - dropoff: (ed - 30,                ld + max(ds_accepted))

    where min(ps_accepted) ≤ 0 is the user's max early-advance for pickup,
    and max(ds_accepted) ≥ 0 is the user's max late-delay for dropoff.
    """
    td2 = td.clone()
    tw = td2["time_windows"].clone()

    for r in range(1, n_customers + 1):
        p, d = 2 * r - 1, 2 * r
        accepted = accepted_per_request.get(r, [(12, 0, 0)])

        # User-preference-limited directions
        max_early_pickup = min(a[1] for a in accepted)   # most negative ps (e.g. -30)
        max_late_dropoff = max(a[2] for a in accepted)   # largest ds (e.g. +30)

        # Always-permissible directions: +30 for delay pickup, -30 for advance dropoff
        tw[0, p, 0] += max_early_pickup   # earliest pickup shifts earlier (negative)
        tw[0, p, 1] += 30                 # latest pickup shifts later (always ok)
        tw[0, d, 0] -= 30                 # earliest dropoff shifts earlier (always ok)
        tw[0, d, 1] += max_late_dropoff   # latest dropoff shifts later (user-limited)

    tw = torch.clamp(tw, min=30, max=1410)
    td2["time_windows"] = tw
    return _prepare_td(td2)


# ===================================================================
# 4.  Retroactive action selection (rounding step)
# ===================================================================

_SHIFT_GRID = np.array(_SHIFT_STEPS_LIST, dtype=float)  # [-30,-20,-10,0,10,20,30]


def _snap_smallest(lo: float, hi: float) -> int:
    """Pick the discrete step in [lo, hi] with smallest absolute value."""
    valid = _SHIFT_GRID[(lo <= _SHIFT_GRID) & (_SHIFT_GRID <= hi)]
    if len(valid) == 0:
        return 0
    return int(valid[np.argmin(np.abs(valid))])


def select_actions(
    visit_times: Dict[int, float],
    tw_original: np.ndarray,
    accepted_per_request: Dict[int, List[Tuple[int, int, int]]],
    n_customers: int,
) -> Dict[int, Tuple[int, int]]:
    """Round relaxed visit times to the smallest discrete (ps, ds) shift.

    For each request, the valid range is computed independently:
      ps ∈ [B_p - l_p,  B_p - e_p]   (shifted window must contain B_p)
      ds ∈ [B_d - l_d,  B_d - e_d]   (shifted window must contain B_d)

    User preferences limit early pickup (ps<0) and late dropoff (ds>0);
    delay pickup (ps>0) and advance dropoff (ds<0) are always permissible.
    """
    selected: Dict[int, Tuple[int, int]] = {}

    for r in range(1, n_customers + 1):
        p, d = 2 * r - 1, 2 * r
        B_p = visit_times.get(p)
        B_d = visit_times.get(d)

        if B_p is None or B_d is None:
            selected[r] = (0, 0)
            continue

        e_p, l_p = float(tw_original[p, 0]), float(tw_original[p, 1])
        e_d, l_d = float(tw_original[d, 0]), float(tw_original[d, 1])

        accepted = accepted_per_request.get(r, [(12, 0, 0)])
        user_min_ps = min(a[1] for a in accepted)   # e.g. -30
        user_max_ds = max(a[2] for a in accepted)   # e.g. +30

        # Valid ps range: shifted window [e_p+ps, l_p+ps] must contain B_p
        ps_lo = B_p - l_p
        ps_hi = B_p - e_p
        # User pref limits early pickup (ps<0); delay (ps>0) always ok
        ps_lo = max(ps_lo, user_min_ps)
        # ps>0 always allowed, so ps_hi capped only by grid max (+30)

        # Valid ds range: shifted window [e_d+ds, l_d+ds] must contain B_d
        ds_lo = B_d - l_d
        ds_hi = B_d - e_d
        # User pref limits late dropoff (ds>0); advance (ds<0) always ok
        ds_hi = min(ds_hi, user_max_ds)
        # ds<0 always allowed, so ds_lo capped only by grid min (-30)

        ps = _snap_smallest(ps_lo, ps_hi)
        ds = _snap_smallest(ds_lo, ds_hi)
        selected[r] = (ps, ds)

    return selected


# ===================================================================
# 5.  Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OR-Tools DARP perturbation via relaxation + rounding")
    parser.add_argument("--customers", type=int, default=30)
    parser.add_argument("--vehicles", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=30,
                        help="OR-Tools time limit per solve (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_cust = args.customers
    csv_path = SRC_DIR / "traveler_trip_types_res_7.csv"
    ttm_path = SRC_DIR / "travel_time_matrix_res_7.csv"
    decisions_csv = SRC_DIR / "traveler_decisions_augmented.csv"

    # --- 1. Generate instance ------------------------------------------------
    print(f"Generating {n_cust}-customer instance (seed={args.seed}) ...")
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

    # --- 2. Build accepted actions per request --------------------------------
    print("Loading traveler decisions ...")
    decision_lookup = build_decision_lookup(decisions_csv)

    trip_metadata = td["trip_metadata"][0]
    flex_raw = td["flexibility"][0].cpu().numpy()
    idx_to_flex = {v: k for k, v in FLEX_STR_TO_IDX.items()}

    accepted_per_request: Dict[int, List[Tuple[int, int, int]]] = {}
    for r in range(1, n_cust + 1):
        p_node = 2 * r - 1
        traveler_id = int(user_ids[p_node])
        flex_idx = int(flex_raw[p_node])
        flex_str = idx_to_flex.get(
            flex_idx, "inflexible for any schedule changes")
        meta = trip_metadata.get(traveler_id)
        if meta is None:
            accepted_per_request[r] = [(12, 0, 0)]
            continue
        accepted_per_request[r] = get_accepted_actions(
            traveler_id, meta, decision_lookup, flex_str)

    total_acc = sum(len(v) for v in accepted_per_request.values())
    min_acc = min(len(v) for v in accepted_per_request.values())
    max_acc = max(len(v) for v in accepted_per_request.values())
    print(f"  Accepted actions: min={min_acc}  max={max_acc}  total={total_acc}")

    # --- 3. STEP 1 — Baseline (no perturbation) ------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: BASELINE  (no perturbation)")
    print("=" * 60)

    baseline = darp_solve(
        td, max_vehicles=args.vehicles,
        time_limit_seconds=args.time_limit)

    print(f"  Cost          : {baseline['total_distance']:.2f} min")
    print(f"  Vehicles used : {baseline['vehicles_used']}")
    for v, route in enumerate(baseline["routes"]):
        print(f"    Vehicle {v + 1}  : [0] -> {route} -> [0]")

    # --- 4. STEP 2 — Slacked solve (widened windows) --------------------------
    print("\n" + "=" * 60)
    print("STEP 2: SLACKED SOLVE  (widened time windows)")
    print("=" * 60)

    td_wide = widen_time_windows(td, accepted_per_request, n_cust)
    tw_wide = td_wide["time_windows"][0].cpu().numpy()

    widened = 0
    for r in range(1, n_cust + 1):
        p, d = 2 * r - 1, 2 * r
        if tw[p, 0] != tw_wide[p, 0] or tw[d, 1] != tw_wide[d, 1]:
            widened += 1
    print(f"  Requests with widened windows : {widened}/{n_cust}")

    slacked = darp_solve(
        td_wide, max_vehicles=args.vehicles,
        time_limit_seconds=args.time_limit)

    print(f"  Cost (lower bound)           : {slacked['total_distance']:.2f} min")
    print(f"  Vehicles used                : {slacked['vehicles_used']}")
    for v, route in enumerate(slacked["routes"]):
        print(f"    Vehicle {v + 1}  : [0] -> {route} -> [0]")

    # --- 5. STEP 3 — Retroactive action selection -----------------------------
    print("\n" + "=" * 60)
    print("STEP 3: RETROACTIVE ACTION SELECTION")
    print("=" * 60)

    selected = select_actions(
        slacked["visit_times"], tw, accepted_per_request, n_cust)

    perturbed_count = sum(1 for s in selected.values() if s != (0, 0))
    print(f"  Requests perturbed : {perturbed_count}/{n_cust}")

    header = (f"{'Req':>4} {'Trav':>5} {'PShift':>7} "
              f"{'DShift':>7}  {'Pickup TW':>20}  {'Delivery TW':>20}"
              f"  {'B_p':>8} {'B_d':>8}")
    print(f"\n{header}")
    print("-" * len(header))
    for r in range(1, n_cust + 1):
        p, d = 2 * r - 1, 2 * r
        ps, ds = selected[r]
        trav = int(user_ids[p])
        bp = slacked["visit_times"].get(p, float("nan"))
        bd = slacked["visit_times"].get(d, float("nan"))
        print(
            f"{r:4d} {trav:5d} {ps:+5d}m  {ds:+5d}m   "
            f"[{tw[p, 0] + ps:6.0f}, {tw[p, 1] + ps:6.0f}]        "
            f"[{tw[d, 0] + ds:6.0f}, {tw[d, 1] + ds:6.0f}]    "
            f"{bp:8.1f} {bd:8.1f}")

    # --- 6. STEP 4 — Re-solve with selected discrete actions ------------------
    print("\n" + "=" * 60)
    print("STEP 4: RE-SOLVE WITH SELECTED ACTIONS")
    print("=" * 60)

    td_pert = _apply_actions(td, selected)
    perturbed = darp_solve(
        td_pert, max_vehicles=args.vehicles,
        time_limit_seconds=args.time_limit)

    print(f"  Cost          : {perturbed['total_distance']:.2f} min")
    print(f"  Vehicles used : {perturbed['vehicles_used']}")
    for v, route in enumerate(perturbed["routes"]):
        print(f"    Vehicle {v + 1}  : [0] -> {route} -> [0]")

    # --- 7. Summary -----------------------------------------------------------
    base_cost = baseline["total_distance"]
    slack_cost = slacked["total_distance"]
    pert_cost = perturbed["total_distance"]

    savings = base_cost - pert_cost
    pct = 100 * savings / base_cost if base_cost > 0 else 0
    lb_savings = base_cost - slack_cost
    lb_pct = 100 * lb_savings / base_cost if base_cost > 0 else 0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline cost         : {base_cost:>10.2f} min  "
          f"({baseline['vehicles_used']} vehicles)")
    print(f"  Slacked cost (LB)     : {slack_cost:>10.2f} min  "
          f"({slacked['vehicles_used']} vehicles)")
    print(f"  Perturbed cost        : {pert_cost:>10.2f} min  "
          f"({perturbed['vehicles_used']} vehicles)")
    print(f"  LB savings            : {lb_savings:>10.2f} min  ({lb_pct:.2f}%)")
    print(f"  Actual savings        : {savings:>10.2f} min  ({pct:.2f}%)")
    if lb_savings > 0:
        gap_closed = 100 * savings / lb_savings
        print(f"  Gap closed            : {gap_closed:>10.2f}%  "
              f"(actual / LB savings)")


if __name__ == "__main__":
    main()
