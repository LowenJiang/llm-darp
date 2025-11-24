from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Constants
TIME_SCALE = 100  # Scale factor to convert float time to int for OR-Tools


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
    2. Optimization objective is minimizing total travel time.
    3. Travel time is queried from travel_time_matrix using H3 indices.
    4. Service times are 0 (only travel time is considered).

    Args:
        td: TensorDict containing:
            - h3_indices: [batch_size, num_nodes] H3 cell indices for each node
            - travel_time_matrix: [batch_size, num_h3_cells, num_h3_cells] travel times in minutes
            - time_windows: [batch_size, num_nodes, 2] [earliest, latest] in minutes
            - demand: [batch_size, num_nodes] load amounts (positive for pickup, negative for delivery)
            - vehicle_capacity: [batch_size, 1] or int
        max_vehicles: Maximum number of vehicles allowed.
        time_limit_seconds: Time limit for the solver.
        run_until_solution: If True, stops after finding the first feasible solution.

    Returns:
        Dictionary containing:
            - total_distance: Total travel time (minutes).
            - vehicles_used: Number of vehicles used.
            - routes: List of routes (lists of node indices).
    """
    # --- 1. Data Extraction & Preprocessing ---
    # We assume batch_size=1 or we take the first element of the batch
    batch_idx = 0

    # Extract H3 indices and travel time matrix
    h3_indices_tensor = td.get("h3_indices")
    travel_time_matrix_tensor = td.get("travel_time_matrix")
    
    if h3_indices_tensor is None or travel_time_matrix_tensor is None:
        raise ValueError("TensorDict must contain 'h3_indices' and 'travel_time_matrix' fields.")
    
    h3_indices = h3_indices_tensor[batch_idx].detach().cpu().numpy().astype(int)
    num_nodes = h3_indices.shape[0]
    
    # Extract Time Windows
    time_windows = td["time_windows"][batch_idx].detach().cpu().numpy()
    
    # Extract Capacity
    vehicle_capacity_tensor = td["vehicle_capacity"][batch_idx]
    if isinstance(vehicle_capacity_tensor, torch.Tensor):
        vehicle_capacity = int(vehicle_capacity_tensor.reshape(-1)[0].item())
    else:
        vehicle_capacity = int(vehicle_capacity_tensor)

    # --- 2. Build Travel Time Matrix ---
    # Extract global matrix
    global_matrix = travel_time_matrix_tensor[batch_idx].detach().cpu().numpy()
    
    # Select submatrix: local_matrix[i, j] = global_matrix[h3_indices[i], h3_indices[j]]
    travel_time_float = global_matrix[np.ix_(h3_indices, h3_indices)]  # minutes

    # Time matrix for OR-Tools (scaled to int)
    travel_time_matrix = np.rint(travel_time_float * TIME_SCALE).astype(int)
    travel_time_matrix = np.maximum(travel_time_matrix, 0)
    
    # Use travel time as the distance metric (for objective function)
    distance_matrix = travel_time_float

    # --- 3. OR-Tools Model Setup ---
    manager = pywrapcp.RoutingIndexManager(num_nodes, max_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # --- 4. Constraints ---

    # A. Travel Time Cost (Objective)
    # We use travel time (in scaled units) as the cost to minimize
    def time_cost_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(travel_time_matrix[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(time_cost_callback)
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
    # Process demand - in the current schema, demand already includes depot (index 0)
    # and has positive values for pickups, negative for deliveries
    raw_demand = td["demand"][batch_idx].detach().cpu().numpy()
    
    # Convert to int for OR-Tools
    effective_demand = np.rint(raw_demand).astype(int)
    
    # Calculate number of pickup-delivery pairs
    num_requests = (num_nodes - 1) // 2

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
            "total_time": float('inf'),
            "vehicles_used": 0,
            "routes": []
        }

    routes = []
    total_time = 0.0
    vehicles_used = 0

    for vehicle_id in range(max_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        
        # Check if vehicle is used (if next node is not End)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
            
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            # Don't add start depot to route list
            # Route contains only pickup and delivery nodes
            if node_index != 0: 
                route.append(node_index)
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            # Add travel time (in minutes)
            time = distance_matrix[manager.IndexToNode(previous_index), manager.IndexToNode(index)]
            total_time += time

        if route:
            routes.append(route)
            vehicles_used += 1

    return {
        "total_time": total_time,
        "vehicles_used": vehicles_used,
        "routes": routes,
    }



