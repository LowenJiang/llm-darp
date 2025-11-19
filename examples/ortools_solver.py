from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Constants
EARTH_RADIUS_KM = 6371.0
TIME_SCALE = 100      # Scale factor to convert float time to int for OR-Tools
DISTANCE_SCALE = 1000 # Scale factor to convert float distance to int for OR-Tools


def darp_solver(
    td: Any,
    max_vehicles: int = 5,
    time_limit_seconds: int = 30,
    run_until_solution: bool = False,
) -> Dict[str, Any]:
    """
    Solve a Dial-a-Ride Problem (DARP) instance using Google OR-Tools.
    
    Assumptions:
    1. Nodes are ordered: [Depot, Pickup_1, Delivery_1, Pickup_2, Delivery_2, ...].
    2. Optimization objective is minimizing total distance.
    3. Distance is calculated using the Haversine formula (lat/lon).
    4. Service times are 0 (only travel time is considered).

    Args:
        td: TensorDict containing:
            - depot: [batch_size, 2] (lat, lon)
            - locs: [batch_size, num_nodes, 2] (lat, lon). 
            - time_windows: [batch_size, num_nodes, 2] [earliest, latest]
            - demand: [batch_size, num_customers] load amount. 
            - vehicle_capacity: [batch_size, 1] or int
            - vehicle_speed: [batch_size, 1] (km/min)
        max_vehicles: Maximum number of vehicles allowed.
        time_limit_seconds: Time limit for the solver.
        run_until_solution: If True, stops after finding the first feasible solution.

    Returns:
        Dictionary containing:
            - total_distance: Total distance traveled (km).
            - vehicles_used: Number of vehicles used.
            - routes: List of routes (lists of node indices).
    """
    # --- 1. Data Extraction & Preprocessing ---
    # We assume batch_size=1 or we take the first element of the batch
    batch_idx = 0

    # Extract Depot and Locations
    depot = td["depot"][batch_idx].detach().cpu().numpy()  # [2]
    locs = td["locs"][batch_idx].detach().cpu().numpy()    # [num_nodes, 2]

    # Handle case where locs might or might not include depot at index 0
    if np.allclose(locs[0], depot, atol=1e-5):
        all_locs = locs
    else:
        all_locs = np.vstack([depot.reshape(1, 2), locs])

    num_nodes = all_locs.shape[0]
    
    # Extract Time Windows
    time_windows = td["time_windows"][batch_idx].detach().cpu().numpy()
    if time_windows.shape[0] != num_nodes:
        # Adjust if time_windows didn't include depot but logic added it
        if time_windows.shape[0] == num_nodes - 1:
             # Add default full day window for depot if missing
             depot_tw = np.array([[0.0, 24 * 60.0]]) 
             time_windows = np.vstack([depot_tw, time_windows])
    
    # Extract Capacity and Speed
    vehicle_capacity_tensor = td["vehicle_capacity"][batch_idx]
    if isinstance(vehicle_capacity_tensor, torch.Tensor):
        vehicle_capacity = int(vehicle_capacity_tensor.reshape(-1)[0].item())
    else:
        vehicle_capacity = int(vehicle_capacity_tensor)

    vehicle_speed_tensor = td.get("vehicle_speed", None)
    if vehicle_speed_tensor is not None:
        vehicle_speed = float(vehicle_speed_tensor[batch_idx].reshape(-1)[0].item())
    else:
        vehicle_speed = 1.0  # Default

    # --- 2. Matrices Calculation ---
    distance_matrix = _compute_haversine_distance_matrix(all_locs)
    
    # Time matrix (minutes * TIME_SCALE)
    # Note: Service time is effectively 0, so time cost is purely travel time
    travel_time_matrix = (distance_matrix / vehicle_speed) * TIME_SCALE
    travel_time_matrix = np.rint(travel_time_matrix).astype(int)
    travel_time_matrix = np.maximum(travel_time_matrix, 0)

    # --- 3. OR-Tools Model Setup ---
    manager = pywrapcp.RoutingIndexManager(num_nodes, max_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # --- 4. Constraints ---

    # A. Distance Cost (Objective)
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node] * DISTANCE_SCALE)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # B. Time Windows Constraint
    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Returns travel time. Service time is 0.
        return int(travel_time_matrix[from_node, to_node])

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Calculate horizon (max possible time)
    time_windows_scaled = (time_windows * TIME_SCALE).astype(int)
    horizon = int(time_windows_scaled[:, 1].max() + 60 * TIME_SCALE)

    routing.AddDimension(
        time_callback_index,
        horizon,  # Allow waiting time
        horizon,  # Max vehicle travel time
        False,    # Don't force start cumul to zero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Apply Time Windows
    for node in range(num_nodes):
        index = manager.NodeToIndex(node)
        start = int(time_windows_scaled[node, 0])
        end = int(time_windows_scaled[node, 1])
        time_dimension.CumulVar(index).SetRange(start, end)

    # C. Capacity Constraint & DARP Logic
    # Process demand to create correct +/- flows
    # Assumption: Input demand is positive magnitude.
    # Order: Depot, P1, D1, P2, D2...
    raw_demand = td["demand"][batch_idx].detach().cpu().numpy()
    
    # Prepare effective demand array including depot
    effective_demand = np.zeros(num_nodes, dtype=int)
    
    # Logic: Node 1 is P1 (idx 0 in raw), Node 2 is D1 (idx 1 in raw)
    # If raw demand is [1, 1], Node 1 gets +1, Node 2 gets -1.
    num_requests = (num_nodes - 1) // 2
    
    for i in range(num_requests):
        pickup_node = 2 * i + 1
        delivery_node = 2 * i + 2
        
        # Map to raw demand index (offset by 1 because raw usually doesn't have depot)
        # If raw_demand has size num_nodes-1:
        p_demand = int(raw_demand[pickup_node - 1])
        
        effective_demand[pickup_node] = p_demand
        effective_demand[delivery_node] = -p_demand

    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return int(effective_demand[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [vehicle_capacity] * max_vehicles,
        True,  # start cumul to zero
        "Capacity"
    )

    # D. Precedence Constraints (Pickup & Delivery)
    solver = routing.solver()
    for i in range(num_requests):
        pickup_node = 2 * i + 1
        delivery_node = 2 * i + 2
        
        pickup_index = manager.NodeToIndex(pickup_node)
        delivery_index = manager.NodeToIndex(delivery_node)
        
        # 1. Same vehicle must perform both
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        solver.Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        
        # 2. Pickup before Delivery
        solver.Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))

    # --- 5. Solve ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    
    if run_until_solution:
        search_parameters.time_limit.seconds = 3600
        search_parameters.solution_limit = 1
    else:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds

    search_parameters.log_search = False
    solution = routing.SolveWithParameters(search_parameters)

    # --- 6. Result Extraction ---
    if solution is None:
        return {
            "total_distance": float('inf'),
            "vehicles_used": 0,
            "routes": []
        }

    routes = []
    total_distance = 0.0
    vehicles_used = 0

    for vehicle_id in range(max_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        
        # Check if vehicle is used (if next node is not End)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
            
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            # Don't add start depot to list yet if prefer clean output, 
            # but standard VRPTW output often omits start/end depot in the list 
            # or keeps them. Here we keep non-depot nodes.
            if node_index != 0: 
                route.append(node_index)
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            # Add distance
            dist = distance_matrix[manager.IndexToNode(previous_index), manager.IndexToNode(index)]
            total_distance += dist

        if route:
            routes.append(route)
            vehicles_used += 1

    return {
        "total_distance": total_distance,
        "vehicles_used": vehicles_used,
        "routes": routes,
    }


def _compute_haversine_distance_matrix(locs: np.ndarray) -> np.ndarray:
    """
    Compute Haversine distance matrix for GPS coordinates (lat, lon) in degrees.
    Returns distance in Kilometers.
    """
    n = locs.shape[0]
    distances = np.zeros((n, n))
    locs_rad = np.deg2rad(locs)

    lat = locs_rad[:, 0]
    lon = locs_rad[:, 1]

    # Vectorized Haversine
    # diff shape: (n, n)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat / 2)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distances = EARTH_RADIUS_KM * c
    
    # Ensure diagonal is zero
    np.fill_diagonal(distances, 0.0)
    
    return distances


# Example usage for testing
if __name__ == "__main__":
    from tensordict import TensorDict
    
    # Test Data Creation
    td_test = TensorDict(
        {
            "depot": torch.tensor([[37.783333, -122.416667]], dtype=torch.float32),
            # Interleaved: Depot, P1, D1, P2, D2, P3, D3
            "locs": torch.tensor([[
                [37.783333, -122.416667], # Depot (Node 0)
                [37.790, -122.420],       # P1 (Node 1)
                [37.780, -122.410],       # D1 (Node 2)
                [37.795, -122.425],       # P2 (Node 3)
                [37.775, -122.415],       # D2 (Node 4)
                [37.785, -122.430],       # P3 (Node 5)
                [37.800, -122.405],       # D3 (Node 6)
            ]], dtype=torch.float32),
            "time_windows": torch.tensor([[
                [0, 1440],    # Depot
                [480, 540],   # P1
                [500, 600],   # D1
                [520, 620],   # P2
                [540, 640],   # D2
                [560, 660],   # P3
                [580, 680],   # D3
            ]], dtype=torch.float32),
            # Demand magnitude (load size)
            "demand": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.float32),
            "vehicle_capacity": torch.tensor([[4]], dtype=torch.int64),
            "vehicle_speed": torch.tensor([[1.0]], dtype=torch.float32),
        },
        batch_size=torch.Size([1]),
    )

    print("Running DARP Solver...")
    result = darp_solver(td_test, max_vehicles=2, time_limit_seconds=5)
    print(f"Status: Found {result['vehicles_used']} routes.")
    print(f"Total Distance: {result['total_distance']:.3f} km")
    for i, r in enumerate(result['routes']):
        print(f"Route {i+1}: {r}")