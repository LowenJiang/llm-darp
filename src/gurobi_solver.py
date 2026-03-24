"""
Exact DARP solver using Gurobi MILP formulation.

Formulation based on Cordeau (2006) 3-index model, adapted to the
project's TensorDict data representation with H3-indexed travel times.

Node layout:
    0        = depot (start & end)
    2i-1     = pickup for request i,   i = 1..n
    2i       = delivery for request i, i = 1..n
    2n+1     = depot copy (virtual end depot, internal to model)

Decision variables:
    x[i,j,k] ∈ {0,1}  — vehicle k traverses arc (i,j)
    t[i,k]   ≥ 0       — time vehicle k begins service at node i
    q[i,k]   ≥ 0       — load of vehicle k after visiting node i

Objective:
    min  Σ_{k} Σ_{(i,j)} c_{ij} * x[i,j,k]  +  vehicle_cost * Σ_k y_k

where y_k = 1 if vehicle k is used.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise ImportError(
        "gurobipy is required. Install with: pip install gurobipy"
    )


def routes_to_actions(routes: List[List[int]]) -> List[int]:
    """Convert per-vehicle routes into a single action sequence for the policy."""
    actions: List[int] = []
    for route in routes:
        if not route:
            continue
        actions.extend(route)
        actions.append(0)  # return to depot between vehicles
    if not actions:
        actions = [0]
    return actions


def darp_gurobi_solver(
    td: Any,
    max_vehicles: int = 5,
    time_limit_seconds: float = 300.0,
    vehicle_cost: float = 250.0,
    mip_gap: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Solve a DARP instance to optimality using Gurobi.

    Uses a 3-index arc-based MILP formulation with:
      - Pickup-delivery pairing & precedence
      - Vehicle capacity
      - Time windows
      - Fixed vehicle cost

    Args:
        td: TensorDict with the same schema as the OR-Tools solver.
        max_vehicles: Maximum number of vehicles (K).
        time_limit_seconds: Gurobi time limit.
        vehicle_cost: Fixed cost per vehicle used.
        mip_gap: Optimality gap tolerance (0.0 = prove optimality).
        verbose: Whether to print Gurobi log.

    Returns:
        Dictionary matching the OR-Tools solver output format.
    """
    # ------------------------------------------------------------------ #
    #  1. Data extraction (identical to OR-Tools version)
    # ------------------------------------------------------------------ #
    batch_idx = 0

    h3_indices_tensor = td.get("h3_indices")
    travel_time_matrix_tensor = td.get("travel_time_matrix")

    if h3_indices_tensor is None or travel_time_matrix_tensor is None:
        raise ValueError(
            "TensorDict must contain 'h3_indices' and 'travel_time_matrix'."
        )

    h3_indices = h3_indices_tensor[batch_idx].detach().cpu().numpy().astype(int)
    num_nodes_original = h3_indices.shape[0]  # 0, P1, D1, ..., Pn, Dn

    time_windows_raw = td["time_windows"][batch_idx].detach().cpu().numpy()

    vehicle_capacity_tensor = td["vehicle_capacity"][batch_idx]
    if isinstance(vehicle_capacity_tensor, torch.Tensor):
        vehicle_capacity = int(vehicle_capacity_tensor.reshape(-1)[0].item())
    else:
        vehicle_capacity = int(vehicle_capacity_tensor)

    raw_demand = td["demand"][batch_idx].detach().cpu().numpy()
    demand = np.rint(raw_demand).astype(int)

    # Build local travel-time matrix (float, minutes)
    global_matrix = travel_time_matrix_tensor[batch_idx].detach().cpu().numpy()
    cost_orig = global_matrix[np.ix_(h3_indices, h3_indices)]  # [N_orig x N_orig]

    num_requests = (num_nodes_original - 1) // 2
    K = max_vehicles

    # ------------------------------------------------------------------ #
    #  2. Augmented node set: add virtual end depot (node 2n+1)
    # ------------------------------------------------------------------ #
    #  Nodes: 0 = start depot, 1..2n = PD nodes, 2n+1 = end depot
    N = num_nodes_original + 1  # total nodes including virtual end depot
    end_depot = num_nodes_original  # index 2n+1

    # Extend cost matrix: trips to/from end depot cost the same as depot 0
    cost = np.zeros((N, N), dtype=np.float64)
    cost[:num_nodes_original, :num_nodes_original] = cost_orig
    # arcs to end depot = arcs to depot 0
    cost[:num_nodes_original, end_depot] = cost_orig[:, 0]
    # arcs from end depot (shouldn't be used, but set for completeness)
    cost[end_depot, :num_nodes_original] = cost_orig[0, :]
    cost[end_depot, end_depot] = 0.0

    # Extend time windows
    tw = np.zeros((N, 2), dtype=np.float64)
    tw[:num_nodes_original] = time_windows_raw
    tw[end_depot] = time_windows_raw[0]  # same as depot

    # Extend demand
    dem = np.zeros(N, dtype=int)
    dem[:num_nodes_original] = demand
    dem[end_depot] = 0

    # Node sets
    P = list(range(1, 2 * num_requests + 1, 2))   # pickup nodes
    D = list(range(2, 2 * num_requests + 1, 2))    # delivery nodes
    PD = P + D                                       # all request nodes
    ALL = list(range(N))
    vehicles = list(range(K))

    # Big-M for time propagation
    M_time = float(tw[:, 1].max()) + cost.max() + 100.0

    # ------------------------------------------------------------------ #
    #  3. Build Gurobi Model
    # ------------------------------------------------------------------ #
    model = gp.Model("DARP")
    model.Params.TimeLimit = time_limit_seconds
    model.Params.MIPGap = mip_gap
    if not verbose:
        model.Params.OutputFlag = 0

    # --- Decision variables ---
    # x[i,j,k]: binary, vehicle k travels arc (i,j)
    # Only create arcs that make sense (skip self-loops, etc.)
    valid_arcs = []
    for i in ALL:
        for j in ALL:
            if i == j:
                continue
            if i == end_depot:
                continue  # no arcs leaving end depot
            if j == 0:
                continue  # no arcs entering start depot (use end_depot instead)
            # Pickup i cannot go directly to its own delivery's pickup partner
            # (this is implicit, not a hard constraint on arc existence)
            valid_arcs.append((i, j))

    x = {}
    for i, j in valid_arcs:
        for k in vehicles:
            x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    # t[i,k]: continuous, service start time at node i for vehicle k
    t = {}
    for i in ALL:
        for k in vehicles:
            t[i, k] = model.addVar(
                lb=tw[i, 0], ub=tw[i, 1],
                vtype=GRB.CONTINUOUS,
                name=f"t_{i}_{k}",
            )

    # q[i,k]: continuous, vehicle load after visiting node i
    q = {}
    for i in ALL:
        for k in vehicles:
            q[i, k] = model.addVar(
                lb=0, ub=vehicle_capacity,
                vtype=GRB.CONTINUOUS,
                name=f"q_{i}_{k}",
            )

    # y[k]: binary, whether vehicle k is used
    y = {}
    for k in vehicles:
        y[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")

    model.update()

    # --- Objective ---
    obj = gp.LinExpr()
    for i, j in valid_arcs:
        for k in vehicles:
            obj += cost[i, j] * x[i, j, k]
    for k in vehicles:
        obj += vehicle_cost * y[k]
    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints ---

    # (C1) Each request is served exactly once:
    # For each pickup node p_i, exactly one vehicle visits it.
    for p in P:
        model.addConstr(
            gp.quicksum(
                x[p, j, k]
                for k in vehicles
                for j in ALL
                if (p, j, k) in x
            ) == 1,
            name=f"serve_pickup_{p}",
        )

    # (C2) For each request, same vehicle does pickup and delivery:
    # flow conservation at each PD node for each vehicle
    for k in vehicles:
        for i in PD:
            model.addConstr(
                gp.quicksum(x[j, i, k] for j in ALL if (j, i, k) in x)
                - gp.quicksum(x[i, j, k] for j in ALL if (i, j, k) in x)
                == 0,
                name=f"flow_{i}_{k}",
            )

    # (C3) Each vehicle starts at depot 0 at most once
    for k in vehicles:
        model.addConstr(
            gp.quicksum(x[0, j, k] for j in ALL if (0, j, k) in x) <= 1,
            name=f"start_{k}",
        )
        # Link y[k] to whether vehicle departs depot
        model.addConstr(
            gp.quicksum(x[0, j, k] for j in ALL if (0, j, k) in x) == y[k],
            name=f"use_{k}",
        )

    # (C4) Each vehicle ends at end_depot
    for k in vehicles:
        model.addConstr(
            gp.quicksum(x[i, end_depot, k] for i in ALL if (i, end_depot, k) in x)
            == y[k],
            name=f"end_{k}",
        )

    # (C5) Pickup before delivery (same vehicle): handled via time + pairing
    # We enforce that for request r, pickup 2r-1 and delivery 2r are on the same
    # vehicle. This is already implied by flow conservation + single-visit, but
    # we add explicit linking for clarity:
    for r in range(num_requests):
        p_node = 2 * r + 1
        d_node = 2 * r + 2
        for k in vehicles:
            model.addConstr(
                gp.quicksum(x[j, p_node, k] for j in ALL if (j, p_node, k) in x)
                == gp.quicksum(x[j, d_node, k] for j in ALL if (j, d_node, k) in x),
                name=f"pair_{r}_{k}",
            )

    # (C6) Time consistency: if x[i,j,k]=1 then t[j,k] >= t[i,k] + cost[i,j]
    for i, j in valid_arcs:
        for k in vehicles:
            if (i, j, k) in x:
                model.addConstr(
                    t[j, k] >= t[i, k] + cost[i, j] - M_time * (1 - x[i, j, k]),
                    name=f"time_{i}_{j}_{k}",
                )

    # (C7) Pickup before delivery in time
    for r in range(num_requests):
        p_node = 2 * r + 1
        d_node = 2 * r + 2
        for k in vehicles:
            model.addConstr(
                t[d_node, k] >= t[p_node, k] + cost[p_node, d_node]
                - M_time * (1 - gp.quicksum(
                    x[j, p_node, k] for j in ALL if (j, p_node, k) in x
                )),
                name=f"prec_{r}_{k}",
            )

    # (C8) Load consistency: if x[i,j,k]=1 then q[j,k] >= q[i,k] + dem[j]
    M_load = vehicle_capacity + max(abs(dem.min()), abs(dem.max())) + 1
    for i, j in valid_arcs:
        if j == end_depot or j == 0:
            continue  # depot load is 0
        for k in vehicles:
            if (i, j, k) in x:
                model.addConstr(
                    q[j, k] >= q[i, k] + dem[j] - M_load * (1 - x[i, j, k]),
                    name=f"loadL_{i}_{j}_{k}",
                )
                model.addConstr(
                    q[j, k] <= q[i, k] + dem[j] + M_load * (1 - x[i, j, k]),
                    name=f"loadU_{i}_{j}_{k}",
                )

    # (C9) Load at depot is 0
    for k in vehicles:
        model.addConstr(q[0, k] == 0, name=f"load_depot_start_{k}")
        model.addConstr(q[end_depot, k] == 0, name=f"load_depot_end_{k}")

    # ------------------------------------------------------------------ #
    #  4. Solve
    # ------------------------------------------------------------------ #
    model.optimize()

    # ------------------------------------------------------------------ #
    #  5. Extract solution
    # ------------------------------------------------------------------ #
    inf_result = {
        "total_time": float("inf"),
        "total_distance": float("inf"),
        "total_cost": float("inf"),
        "vehicles_used": 0,
        "routes": [],
        "actions": [],
        "actions_tensor": torch.tensor([], dtype=torch.long),
        "mip_gap": float("inf"),
        "status": "INFEASIBLE",
    }

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        return inf_result

    if model.SolCount == 0:
        return inf_result

    # Reconstruct routes
    routes = []
    total_distance = 0.0
    vehicles_used = 0

    for k in vehicles:
        if y[k].X < 0.5:
            continue

        # Build adjacency for vehicle k
        succ = {}
        for i, j in valid_arcs:
            if (i, j, k) in x and x[i, j, k].X > 0.5:
                succ[i] = j

        # Trace route from depot 0
        route = []
        current = 0
        visited = {0}
        while current in succ:
            nxt = succ[current]
            if nxt == end_depot:
                # Add travel time for last arc
                total_distance += cost[current, 0]  # map end_depot back to depot 0
                break
            total_distance += cost[current, nxt]
            route.append(nxt)
            visited.add(nxt)
            current = nxt
            if len(visited) > N:
                break  # safety: avoid infinite loops

        if route:
            routes.append(route)
            vehicles_used += 1

    actions = routes_to_actions(routes)
    total_cost = total_distance + vehicle_cost * vehicles_used

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }

    return {
        "total_time": total_distance,
        "total_distance": total_distance,
        "total_cost": total_cost,
        "vehicles_used": vehicles_used,
        "routes": routes,
        "actions": actions,
        "actions_tensor": torch.tensor(actions, dtype=torch.long),
        "mip_gap": model.MIPGap if model.SolCount > 0 else float("inf"),
        "status": status_map.get(model.Status, f"STATUS_{model.Status}"),
        "objective_bound": model.ObjBound if model.SolCount > 0 else float("inf"),
    }


# ---------------------------------------------------------------------- #
#  Main: test on the same generated instance as the OR-Tools solver
# ---------------------------------------------------------------------- #
def main() -> None:
    """Solve one generated instance with both OR-Tools and Gurobi, compare."""
    from pathlib import Path

    # Import project modules (same as oracle_ortools.py main)
    from oracle_env import PDPTWEnv
    from oracle_generator import SFGenerator

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")

    generator = SFGenerator(csv_path=csv_path, travel_time_matrix_path=ttm_path)
    env = PDPTWEnv(generator=generator)

    batch = generator(batch_size=[1]).to(device)
    state = env.reset(batch)

    num_requests = (state["h3_indices"].shape[-1] - 1) // 2
    print(f"Instance: {num_requests} requests")
    print("=" * 60)

    # --- OR-Tools baseline ---
    try:
        from or_tools import darp_solver

        print("Solving with OR-Tools (30s)...")
        ort_result = darp_solver(state, max_vehicles=5, time_limit_seconds=30)
        print(f"  Distance : {ort_result['total_distance']:.2f}")
        print(f"  Vehicles : {ort_result['vehicles_used']}")
        print(f"  Cost     : {ort_result['total_cost']:.2f}")
        print(f"  Routes   : {ort_result['routes']}")
    except ImportError:
        print("OR-Tools solver not available, skipping baseline.")
        ort_result = None

    print()

    # --- Gurobi exact ---
    print("Solving with Gurobi (300s, exact)...")
    gurobi_result = darp_gurobi_solver(
        state,
        max_vehicles=5,
        time_limit_seconds=300,
        vehicle_cost=250.0,
        mip_gap=0.0,
        verbose=True,
    )
    print(f"  Status   : {gurobi_result['status']}")
    print(f"  Distance : {gurobi_result['total_distance']:.2f}")
    print(f"  Vehicles : {gurobi_result['vehicles_used']}")
    print(f"  Cost     : {gurobi_result['total_cost']:.2f}")
    print(f"  MIP Gap  : {gurobi_result['mip_gap']:.6f}")
    print(f"  ObjBound : {gurobi_result.get('objective_bound', 'N/A')}")
    print(f"  Routes   : {gurobi_result['routes']}")

    # --- Comparison ---
    if ort_result and ort_result["total_cost"] < float("inf"):
        print()
        print("=" * 60)
        gap = (
            (ort_result["total_cost"] - gurobi_result["total_cost"])
            / gurobi_result["total_cost"]
            * 100
            if gurobi_result["total_cost"] > 0
            else float("inf")
        )
        print(f"OR-Tools vs Gurobi cost gap: {gap:+.2f}%")


if __name__ == "__main__":
    main()