from __future__ import annotations
from typing import Optional, Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from pathlib import Path


def gather_by_index(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = 1,
    squeeze: bool = True
) -> torch.Tensor:
    """Gather values from source tensor along dim using index."""
    if index.dim() == 1:
        index = index.unsqueeze(-1)
    
    # Expand index to match source dimensions for gathering
    if dim == 1:
        if source.dim() == index.dim():
            result = source.gather(dim, index)
        else:
            expanded_index = index.unsqueeze(-1).expand(*index.shape, source.shape[-1])
            result = source.gather(dim, expanded_index)
    elif dim == 2:
        if index.dim() == source.dim() - 1:
            index = index.unsqueeze(1).expand(-1, source.shape[1], *index.shape[1:])
        result = source.gather(dim, index)
    else:
        expanded_index = index.expand(*source.shape[:-1], index.shape[-1])
        result = source.gather(dim, expanded_index)
    
    if squeeze and result.shape[-1] == 1:
        result = result.squeeze(-1)
    return result


class PDPTWEnv(nn.Module):
    """
    Pickup and Delivery Problem with Time Windows (PDPTW) Environment.
    
    A vehicle routing environment where:
    - Vehicles must pick up items and deliver them to destinations
    - Each pickup has a corresponding delivery (paired nodes)
    - Pickups must occur before deliveries (precedence constraints)
    - All visits must respect time windows
    - Vehicle capacity limits apply
    
    Node indexing:
    - Node 0: Depot
    - Odd nodes (1, 3, 5, ...): Pickup locations
    - Even nodes (2, 4, 6, ...): Corresponding delivery locations
    - Pickup i is paired with delivery i+1
    
    Masking Strategy (one-step lookahead):
    - Time window reachability: arrival must be within the candidate's time window
    - Capacity constraint: vehicle must have room for pickup demand
    - Precedence constraint: pickup must be visited before its delivery
    - Lookahead: visiting candidate must still allow all onboard deliveries
      (and, for pickups, their own delivery) to be reached within time windows
    - Depot allowed when no reachable dropoff exists (replaces forced delivery)
    - Safety valve: if all actions masked, depot is allowed as escape

    Dynamic Visitation Management:
    - Returning to depot starts a new vehicle with time reset to zero; vehicles used are counted
    """
    
    def __init__(
        self,
        generator: Optional[Callable] = None,
        vehicle_capacity: Optional[int] = None,
        vehicle_penalty: float = 50.0,
        free_vehicles: int = 5,
        **kwargs
    ):
        super().__init__()
        self.generator = generator
        if vehicle_capacity is None and hasattr(generator, "vehicle_capacity"):
            vehicle_capacity = getattr(generator, "vehicle_capacity")
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_penalty = float(vehicle_penalty)
        self.free_vehicles = int(free_vehicles)
        
    # =========================================================================
    # Core Environment Interface
    # =========================================================================
    
    def reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None
    ) -> TensorDict:
        """
        Reset environment to initial state.
        
        Args:
            td: Optional TensorDict with problem instance data
            batch_size: Batch dimensions
            
        Returns:
            TensorDict with initial state
        """
        if td is None:
            if self.generator is None:
                raise ValueError("Must provide either td or generator")
            td = self.generator(batch_size=batch_size)
        
        if batch_size is None:
            batch_size = td.batch_size

        # Get device from input tensors
        # Use the device of h3_indices as the canonical source
        h3_device = td["h3_indices"].device
        device = h3_device

        # Normalize device to ensure it has an index
        if device.type in ("mps", "cuda") and device.index is None:
            device = torch.device(f"{device.type}:0")

        # Extract problem data
        h3_indices = td["h3_indices"]
        demands = td["demand"]
        time_windows = td["time_windows"]
        travel_time_matrix = td["travel_time_matrix"]
        locs = td.get("locs", None)
        flexibility = td.get("flexibility", torch.zeros_like(demands))
        user_id = td.get("user_id", None)
        trip_metadata = td.get("trip_metadata", None)
        
        # Initialize state
        capacity_from_td = td.get("capacity", None)
        if capacity_from_td is None:
            if self.vehicle_capacity is None:
                raise ValueError("Vehicle capacity must be provided by generator or env init.")
            capacity_limit = int(self.vehicle_capacity)
            capacity_tensor = torch.full((*batch_size, 1), float(capacity_limit), device=device)
        else:
            capacity_tensor = capacity_from_td.to(device).view(*batch_size, 1)
            capacity_limit = int(capacity_tensor.max().item())
        pending_schedule = torch.zeros(
            (*batch_size, capacity_limit), 
            dtype=torch.int64, 
            device=device
        )
        pending_count = torch.zeros(
            (*batch_size, 1), 
            dtype=torch.int64, 
            device=device
        )
        
        state = TensorDict({
            # Problem data (static)
            "h3_indices": h3_indices,
            "travel_time_matrix": travel_time_matrix,
            "time_windows": time_windows,
            "demand": demands,
            "flexibility": flexibility,
            "vehicle_capacity": capacity_tensor,
            
            # Dynamic state
            "current_node": torch.zeros(
                *batch_size, 1, 
                dtype=torch.int64, 
                device=device
            ),
            "current_time": torch.zeros(
                *batch_size, 1, 
                dtype=torch.float32, 
                device=device
            ),
            "used_capacity": torch.zeros(
                *batch_size, 1, 
                dtype=torch.float32, 
                device=device
            ),
            "visited": torch.zeros(
                *batch_size, h3_indices.shape[-1], 
                dtype=torch.bool, 
                device=device
            ),
            "completed": torch.zeros(
                *batch_size, h3_indices.shape[-1],
                dtype=torch.bool,
                device=device
            ),
            "pending_schedule": pending_schedule,
            "pending_count": pending_count,
            "previous_action": torch.zeros(
                *batch_size, 1, 
                dtype=torch.int64, 
                device=device
            ),
            "i": torch.zeros(
                *batch_size, 1, 
                dtype=torch.int64, 
                device=device
            ),
            "vehicles_used": torch.ones(
                *batch_size, 1,
                dtype=torch.int64,
                device=device
            ),
            "done": torch.zeros(
                *batch_size,
                dtype=torch.bool,
                device=device
            ),
        }, batch_size=batch_size)
        
        if locs is not None:
            state.set("locs", locs)
        if user_id is not None:
            state.set("user_id", user_id)
        if trip_metadata is not None:
            state.set("trip_metadata", trip_metadata)
        
        # Compute initial action mask
        state.set("action_mask", self._compute_action_mask(state))
        
        return state
    
    def step(self, td: TensorDict) -> TensorDict:
        """
        Execute one step in the environment.
        
        Args:
            td: TensorDict with current state and action
            
        Returns:
            TensorDict with updated state
        """
        action = td["action"]
        batch_size = action.shape[0]

        # Get device from input tensors
        h3_device = td["h3_indices"].device
        device = h3_device

        # Normalize device to ensure it has an index
        if device.type in ("mps", "cuda") and device.index is None:
            device = torch.device(f"{device.type}:0")

        curr_node = td["current_node"]
        
        # Calculate arrival time at selected node
        travel_time = self._get_travel_time(td, curr_node, action.unsqueeze(-1))
        arrival_time = td["current_time"] + travel_time
        
        # Service starts at max(arrival, window_start)
        start_window = gather_by_index(td["time_windows"], action)[..., 0]
        service_start_time = torch.max(arrival_time, start_window)
        
        # Handle depot returns - reset time
        is_return_to_depot = (action == 0) & (curr_node.squeeze(-1) != 0)
        service_start_time = torch.where(
            is_return_to_depot.unsqueeze(-1),
            torch.zeros_like(service_start_time),
            service_start_time
        )
        
        # Update capacity
        selected_demand = gather_by_index(td["demand"], action).unsqueeze(-1)
        new_load = td["used_capacity"] + selected_demand
        new_load = torch.where(
            is_return_to_depot.unsqueeze(-1),
            torch.zeros_like(new_load),
            new_load
        )
        
        # Track completed requests (only on successful dropoff)
        new_completed = td["completed"].clone()
        is_pickup = (action % 2 != 0) & (action != 0)
        is_dropoff = (action % 2 == 0) & (action != 0)
        partner_node = torch.clamp(action - 1, min=0)
        mark_complete = torch.ones_like(action, dtype=torch.bool).unsqueeze(-1)
        new_completed = torch.where(
            is_dropoff.unsqueeze(-1),
            new_completed.scatter(-1, action.unsqueeze(-1), mark_complete)
                         .scatter(-1, partner_node.unsqueeze(-1), mark_complete),
            new_completed
        )
        
        # Track visits and reset unresolved pickups on depot return
        new_visited = td["visited"].clone()
        non_depot_mask = (action != 0).unsqueeze(-1)
        new_visited.scatter_(-1, action.unsqueeze(-1), non_depot_mask)
        
        unresolved_mask = (td["pending_schedule"] != 0) & is_return_to_depot.unsqueeze(-1)
        unresolved_pickups = torch.clamp(td["pending_schedule"] - 1, min=0)
        reset_mask = torch.zeros_like(new_visited)
        reset_mask.scatter_(1, unresolved_pickups, unresolved_mask)
        new_visited = new_visited & ~reset_mask
        
        # Update pending schedule (obligations)
        pending_schedule, pending_count = self._update_pending_schedule(
            td, action, is_return_to_depot, batch_size, device
        )
        
        # Vehicle usage tracking
        vehicles_used = td["vehicles_used"] + is_return_to_depot.long().unsqueeze(-1)

        # Build output state — share static tensors, only copy dynamic ones
        td_out = td.empty()  # shallow shell, no tensor copies
        # Static data: share references (never mutated)
        for key in ("h3_indices", "travel_time_matrix", "time_windows",
                     "demand", "flexibility", "vehicle_capacity"):
            td_out.set(key, td[key])
        # Optional static fields
        for key in ("locs", "user_id", "trip_metadata"):
            if key in td.keys():
                td_out.set(key, td[key])
        # Dynamic state: new tensors computed above
        td_out.update({
            "current_node": action.unsqueeze(-1),
            "current_time": service_start_time,
            "used_capacity": new_load,
            "visited": new_visited,
            "completed": new_completed,
            "pending_schedule": pending_schedule,
            "pending_count": pending_count,
            "previous_action": action.unsqueeze(-1),
            "i": td["i"] + 1,
            "vehicles_used": vehicles_used,
        })
        
        # Check termination conditions
        is_depot = (action == 0)
        all_completed = new_completed[..., 1:].all(dim=-1)
        td_out.set("done", is_depot & all_completed)
        td_out.set("action_mask", self._compute_action_mask(td_out))
        
        return TensorDict({"next": td_out}, batch_size=td_out.batch_size)
    
    def get_reward(
        self,
        td: TensorDict,
        actions: torch.Tensor,
        use_vectorized: bool = True,
        raw_cost: bool = False,
    ) -> torch.Tensor:
        """
        Compute reward for completed episode.

        Args:
            td: TensorDict with problem data
            actions: Sequence of actions taken [batch_size, seq_len]
            use_vectorized: Use fast vectorized implementation (default: True)
            raw_cost: If True, compute cost from the full action sequence
                      (including retracted pickups and wasted depot returns).
                      This penalises inefficient exploration directly.

        Returns:
            Reward tensor [batch_size]
        """
        if use_vectorized:
            # Fast vectorized path - all batches in parallel
            if raw_cost:
                valid_mask = torch.ones_like(actions, dtype=torch.bool)
            else:
                valid_mask = self._build_valid_action_mask(actions)
            cost_tensor = self._compute_route_costs_vectorized(td, actions, valid_mask)
        else:
            # Legacy sequential path - kept for validation/debugging
            effective_routes = self._build_effective_routes(actions)

            costs = []
            for batch_idx, route in enumerate(effective_routes):
                cost = self._travel_time_for_route(td, batch_idx, route)
                costs.append(cost)
            cost_tensor = torch.stack(costs)

        td.set("cost", cost_tensor)

        # Vehicle penalty: charge for each vehicle beyond the free allowance
        vehicles_used = td.get(
            "vehicles_used",
            torch.ones_like(cost_tensor).unsqueeze(-1)
        ).squeeze(-1).float()
        excess_vehicles = torch.clamp(vehicles_used - self.free_vehicles, min=0)
        vehicle_penalty = self.vehicle_penalty * excess_vehicles

        reward = -cost_tensor - vehicle_penalty
        return reward
    
    # =========================================================================
    # Travel Time Computation
    # =========================================================================
    
    def _get_travel_time(
        self,
        td: TensorDict,
        from_node_idx: torch.Tensor,
        to_node_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Get travel time between nodes using the travel time matrix.
        
        Args:
            td: TensorDict containing h3_indices and travel_time_matrix
            from_node_idx: Source node indices
            to_node_idx: Destination node indices
            
        Returns:
            Travel times between node pairs
        """
        from_h3 = gather_by_index(td["h3_indices"], from_node_idx, squeeze=False)
        to_h3 = gather_by_index(td["h3_indices"], to_node_idx, squeeze=False)

        rows = gather_by_index(
            td["travel_time_matrix"],
            from_h3,
            dim=1,
            squeeze=False
        )

        if rows.size(1) == 1 and to_h3.size(1) > 1:
            rows = rows.expand(-1, to_h3.size(1), -1)

        travel = rows.gather(2, to_h3.unsqueeze(-1))
        return travel.squeeze(-1)
    
    # =========================================================================
    # Action Masking
    # =========================================================================
    
    def _lookahead_validity(
        self,
        td: TensorDict,
        time_at_candidate: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        One-step lookahead: for each candidate node, check that ALL onboard
        deliveries (in pending_schedule) can still be reached within their late
        time windows after visiting the candidate.  If the candidate is a pickup,
        also check that its corresponding delivery is reachable.

        Args:
            td: Current state TensorDict
            time_at_candidate: [B, N] service-start time at each candidate
            node_indices: [B, N] arange node indices

        Returns:
            valid: [B, N] bool — True if visiting candidate keeps all
                   obligations feasible
        """
        pending_schedule = td["pending_schedule"]       # [B, capacity]
        pending_mask = pending_schedule != 0             # [B, capacity]
        late_windows = td["time_windows"][..., 1]       # [B, N]
        num_nodes = node_indices.shape[1]

        valid = torch.ones_like(node_indices, dtype=torch.bool)  # [B, N]

        # --- Check each onboard delivery ----------------------------------
        for slot in range(pending_schedule.shape[1]):
            target = pending_schedule[:, slot]            # [B]
            active = pending_mask[:, slot]                # [B]
            if not active.any():
                continue

            # travel from candidate -> onboard delivery target
            target_exp = target.unsqueeze(-1).expand_as(node_indices)  # [B, N]
            travel = self._get_travel_time(td, node_indices, target_exp)  # [B, N]
            arrival = time_at_candidate + travel          # [B, N]

            target_late = late_windows.gather(1, target.unsqueeze(-1))  # [B, 1]
            feasible = arrival <= target_late              # [B, N]
            # only constrain batches that actually have this slot occupied
            valid = valid & (feasible | ~active.unsqueeze(-1))

        # --- If candidate is a pickup, also check its delivery reachable ---
        is_pickup = (node_indices % 2 != 0) & (node_indices != 0)
        delivery_node = torch.clamp(node_indices + 1, max=num_nodes - 1)  # [B, N]
        travel_to_delivery = self._get_travel_time(td, node_indices, delivery_node)
        arrival_at_delivery = time_at_candidate + travel_to_delivery
        delivery_late = late_windows.gather(1, delivery_node)
        delivery_feasible = arrival_at_delivery <= delivery_late
        valid = valid & (delivery_feasible | ~is_pickup)

        return valid

    def _compute_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask using time window reachability, capacity,
        precedence, and one-step lookahead validity.

        Lookahead prevents infeasible pickups at the source, removing the
        need for forced delivery or lateness penalties.

        Args:
            td: Current state TensorDict

        Returns:
            Boolean mask [batch_size, num_nodes] - True if action is valid
        """
        batch_size = td["h3_indices"].shape[0]
        num_nodes = td["h3_indices"].shape[1]
        device = td["h3_indices"].device

        current_node = td["current_node"].squeeze(-1)
        current_time = td["current_time"]
        node_indices = torch.arange(num_nodes, device=device).expand(batch_size, -1)

        is_depot = node_indices == 0
        is_pickup = (node_indices % 2 != 0) & ~is_depot
        is_dropoff = (node_indices % 2 == 0) & ~is_depot

        # Track onboard pickups via pending schedule (stores dropoffs)
        pending_schedule = td["pending_schedule"]
        pending_mask = pending_schedule != 0
        onboard_pickups = torch.zeros_like(td["completed"])
        onboard_pickups.scatter_(1, torch.clamp(pending_schedule - 1, min=0), pending_mask)

        # Capacity constraint
        can_pickup = is_pickup & ~td["completed"] & ~onboard_pickups
        remaining_cap = td["vehicle_capacity"] - td["used_capacity"]
        demand_ok = (td["demand"] <= remaining_cap) | ~is_pickup
        can_pickup = can_pickup & demand_ok

        # Precedence constraint
        pickup_indices = torch.clamp(node_indices - 1, min=0)
        pickup_visited = td["visited"].gather(1, pickup_indices)
        can_dropoff = is_dropoff & pickup_visited & ~td["completed"]

        current_not_depot = (td["current_node"].squeeze(-1) != 0).unsqueeze(-1)
        base_mask = (is_depot & current_not_depot) | can_pickup | can_dropoff

        # Time window reachability
        travel_time_to_candidate = self._get_travel_time(
            td,
            current_node.unsqueeze(-1),
            node_indices
        )
        arrival_at_candidate = current_time + travel_time_to_candidate
        candidate_early = td["time_windows"][..., 0]
        candidate_late = td["time_windows"][..., 1]
        time_feasible = arrival_at_candidate <= candidate_late

        # Service starts at max(arrival, early_window)
        time_at_candidate = torch.max(arrival_at_candidate, candidate_early)

        mask = base_mask & time_feasible

        # Identify scheduled dropoffs (deliveries whose pickups are onboard)
        scheduled = torch.zeros_like(mask)                        # [B, N]
        scheduled.scatter_(1, pending_schedule, pending_mask)
        scheduled[..., 0] = False

        # One-step lookahead: prune candidates that make obligations infeasible
        # Exempt scheduled dropoffs — they are mandatory obligations that must
        # be delivered; filtering them causes depot-return loops.
        lookahead_valid = self._lookahead_validity(td, time_at_candidate, node_indices)
        mask = mask & (lookahead_valid | scheduled)

        # Depot rule: allow depot only if no scheduled dropoff is reachable
        reachable_scheduled = (mask & scheduled).any(dim=-1, keepdim=True)
        allow_depot = (~reachable_scheduled & current_not_depot).squeeze(-1)
        mask[..., 0] = allow_depot

        # Safety valve: if no action is valid, allow depot as escape
        no_valid = ~mask.any(dim=-1)
        mask[no_valid, 0] = True

        return mask
    
    # =========================================================================
    # Schedule Management
    # =========================================================================
    
    def _update_pending_schedule(
        self,
        td: TensorDict,
        action: torch.Tensor,
        is_return_to_depot: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the pending delivery schedule based on action.
        
        - Remove visited node from schedule
        - Add partner node for new pickups
        - Clear schedule on depot return
        """
        pending_schedule = td["pending_schedule"].clone()
        pending_count = td["pending_count"].clone()
        capacity = pending_schedule.shape[1]
        num_nodes = td["h3_indices"].shape[1]
        
        is_pickup = (action % 2 != 0) & (action != 0)
        is_dropoff = (action % 2 == 0) & (action != 0)
        
        # Remove visited node from schedule
        mask_in_schedule = pending_schedule == action.unsqueeze(-1)
        pending_schedule[mask_in_schedule] = 0
        
        # Decrease count for dropoffs
        pending_count = torch.clamp(
            pending_count - is_dropoff.long().unsqueeze(-1), 
            min=0
        )
        
        # Add partner (delivery) for pickups
        partner_node = torch.clamp(action + 1, max=num_nodes - 1)
        new_entry = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
        new_entry[is_pickup, 0] = partner_node[is_pickup]
        combined = torch.cat([pending_schedule, new_entry], dim=1)
        
        # Compact schedule (push zeros to end)
        node_vals = combined.clone()
        node_vals[combined == 0] = int(1e9)
        _, sort_idx = torch.sort(node_vals, dim=1)
        sorted_sched = torch.gather(combined, 1, sort_idx)
        pending_schedule = sorted_sched[:, :capacity]
        
        # Update count for pickups
        pending_count = torch.clamp(
            pending_count + is_pickup.long().unsqueeze(-1), 
            max=capacity
        )
        
        # Clear on depot return
        pending_schedule = torch.where(
            is_return_to_depot.unsqueeze(-1),
            torch.zeros_like(pending_schedule),
            pending_schedule
        )
        pending_count = torch.where(
            is_return_to_depot.unsqueeze(-1),
            torch.zeros_like(pending_count),
            pending_count
        )
        
        return pending_schedule, pending_count

    # =========================================================================
    # Reward Helpers
    # =========================================================================

    def _build_valid_action_mask(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Vectorized: Create mask for valid actions (exclude pickups without completed deliveries).

        For each pickup in the sequence, check if its delivery appears before the next depot.
        If not, exclude that pickup (and its orphaned delivery) from cost calculation.

        Args:
            actions: [B, T] action sequence

        Returns:
            mask: [B, T] boolean mask, True for valid actions to include in cost
        """
        B, T = actions.shape

        # Start with all valid
        valid_mask = torch.ones_like(actions, dtype=torch.bool)

        # Identify depot positions
        is_depot = (actions == 0)

        # For each pickup, find if its delivery appears before next depot
        # This requires per-batch processing due to sequential dependencies
        for b in range(B):
            batch_actions = actions[b]
            segment_start = 0

            # Find all depot positions + end of sequence
            depot_positions = torch.where(is_depot[b])[0].tolist() + [T]

            for depot_pos in depot_positions:
                # Process segment [segment_start, depot_pos)
                segment = batch_actions[segment_start:depot_pos]

                # Track which pickups are delivered in this segment
                pickups_seen = []
                delivered_pickups = set()

                for local_t, node in enumerate(segment):
                    node_val = node.item()
                    global_t = segment_start + local_t

                    if node_val % 2 == 1 and node_val != 0:  # Pickup
                        pickups_seen.append((global_t, node_val))
                    elif node_val % 2 == 0 and node_val != 0:  # Dropoff
                        partner = node_val - 1
                        delivered_pickups.add(partner)

                # Mark undelivered pickups as invalid
                for pickup_t, pickup_node in pickups_seen:
                    if pickup_node not in delivered_pickups:
                        valid_mask[b, pickup_t] = False

                segment_start = depot_pos + 1

        return valid_mask

    def _compute_route_costs_vectorized(
        self,
        td: TensorDict,
        actions: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fully-batched travel cost computation — no Python loop over B.

        Strategy: stable-sort valid actions to the front, prepend depot,
        batch-index H3 lookups and the travel-time matrix, then mask-sum.

        Args:
            td: TensorDict with h3_indices and travel_time_matrix
            actions: [B, T] action sequence
            valid_mask: [B, T] mask for valid actions

        Returns:
            costs: [B] total travel time per batch instance
        """
        B = actions.shape[0]
        device = actions.device

        h3_indices = td["h3_indices"]       # [B, N]
        travel_matrix = td["travel_time_matrix"]  # [B, H, H]

        # Number of valid actions per batch element
        valid_counts = valid_mask.sum(dim=1)  # [B]
        max_valid = int(valid_counts.max().item())

        if max_valid == 0:
            return torch.zeros(B, device=device)

        # Compact valid actions to the front via stable descending sort on mask
        _, sort_indices = valid_mask.float().sort(dim=1, descending=True, stable=True)
        compacted = actions.gather(1, sort_indices)[:, :max_valid]  # [B, max_valid]

        # Prepend depot (node 0) to form full route
        depot = torch.zeros(B, 1, dtype=actions.dtype, device=device)
        route = torch.cat([depot, compacted], dim=1)  # [B, max_valid + 1]

        # Map node indices → H3 cell indices
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        route_h3 = h3_indices[batch_idx, route]  # [B, max_valid + 1]

        # Consecutive transition pairs
        from_h3 = route_h3[:, :-1]  # [B, max_valid]
        to_h3   = route_h3[:, 1:]   # [B, max_valid]

        # Transition t is valid when t < valid_counts (depot + t-th action exists)
        trans_idx = torch.arange(max_valid, device=device).unsqueeze(0)  # [1, max_valid]
        trans_mask = trans_idx < valid_counts.unsqueeze(1)  # [B, max_valid]

        # Batch travel-time lookup
        batch_idx_2d = batch_idx.expand_as(from_h3)  # [B, max_valid]
        travel_times = travel_matrix[batch_idx_2d, from_h3, to_h3]  # [B, max_valid]

        # Mask out invalid transitions and sum
        costs = (travel_times * trans_mask.float()).sum(dim=1)  # [B]
        return costs

    def _build_effective_routes(self, actions: torch.Tensor) -> list[list[int]]:
        """
        Remove pickups whose deliveries were not completed before returning to depot.

        NOTE: This is the legacy implementation kept for compatibility.
        The vectorized path uses _build_valid_action_mask instead.
        """
        batch_routes: list[list[int]] = []
        for batch_actions in actions.tolist():
            routes: list[list[int]] = []
            segment_nodes: list[int] = []
            pickups_seen: set[int] = set()
            delivered_pickups: set[int] = set()
            
            for node in batch_actions:
                if node == 0:
                    filtered_segment = [
                        n for n in segment_nodes
                        if not ((n % 2 != 0) and (n not in delivered_pickups))
                    ]
                    if filtered_segment or routes:
                        routes.append([0] + filtered_segment + [0])
                    segment_nodes = []
                    pickups_seen = set()
                    delivered_pickups = set()
                    continue
                
                segment_nodes.append(node)
                if node % 2 != 0:
                    pickups_seen.add(node)
                else:
                    partner = node - 1
                    if partner in pickups_seen:
                        delivered_pickups.add(partner)
            
            if segment_nodes:
                filtered_segment = [
                    n for n in segment_nodes
                    if not ((n % 2 != 0) and (n not in delivered_pickups))
                ]
                routes.append([0] + filtered_segment + [0])
            
            # Flatten segments, keeping depot separators
            if not routes:
                batch_routes.append([0])
            else:
                flattened: list[int] = []
                for idx, seg in enumerate(routes):
                    if idx == 0:
                        flattened.extend(seg)
                    else:
                        flattened.extend(seg[1:])
                batch_routes.append(flattened)
        return batch_routes
    
    def _travel_time_for_route(
        self,
        td: TensorDict,
        batch_idx: int,
        route: list[int]
    ) -> torch.Tensor:
        """Compute travel time for a single effective route."""
        # Get device from actual tensor in td, not from td.device which might be None
        device = td["h3_indices"].device
        if len(route) < 2:
            return torch.tensor(0.0, device=device)

        h3_indices = td["h3_indices"][batch_idx]
        travel_matrix = td["travel_time_matrix"][batch_idx]
        total = travel_matrix.new_tensor(0.0)
        for start, end in zip(route[:-1], route[1:]):
            from_h3 = h3_indices[start]
            to_h3 = h3_indices[end]
            total = total + travel_matrix[from_h3, to_h3]
        return total


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import argparse
    from oracle_generator import SFGenerator

    parser = argparse.ArgumentParser(description="PDPTWEnv test runner")
    parser.add_argument(
        "--policy", type=str, default=None,
        help="Path to a policy checkpoint (.pt). If omitted, uses random actions.",
    )
    parser.add_argument(
        "--decode", type=str, default="greedy", choices=["greedy", "sampling"],
        help="Decode strategy when using a policy (default: greedy).",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-customers", type=int, default=30)
    # Model architecture (must match checkpoint)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-hidden", type=int, default=256)
    args = parser.parse_args()

    batch_size = args.batch_size
    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")

    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        num_customers=args.num_customers
    )

    env = PDPTWEnv(generator=generator)

    # --- Load policy if provided ---
    policy = None
    if args.policy is not None:
        from trial_gnn_policy import DARPPolicy

        policy = DARPPolicy(
            embed_dim=args.embed_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_heads=args.num_heads,
            ff_hidden=args.ff_hidden,
        )
        checkpoint = torch.load(args.policy, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("policy_state_dict", checkpoint)
        policy.load_state_dict(state_dict)
        policy.eval()
        print(f"Loaded policy from {args.policy}  (decode={args.decode})")

    # --- Run with policy (full rollout handled by policy.forward) ---
    if policy is not None:
        td = generator(batch_size=[batch_size])
        with torch.no_grad():
            outputs = policy(
                td,
                env,
                phase="val",
                decode_type=args.decode,
                max_steps=args.max_steps,
                calc_reward=True,
            )

        reward = outputs["reward"]
        actions_tensor = outputs["actions"]
        # Recover final state cost by re-running get_reward on reset state
        state = env.reset(td)
        _ = env.get_reward(state, actions_tensor)
        cost = state["cost"]

    # --- Run with random actions ---
    else:
        state = env.reset(batch_size=[batch_size])
        print(f"Initial action mask counts: {state['action_mask'].sum(dim=-1)}")

        actions_taken = []
        done = torch.zeros(batch_size, dtype=torch.bool)
        for step in range(args.max_steps):
            mask = state["action_mask"]

            # For done envs: set depot as the only valid action to avoid NaN
            safe_mask = mask.clone()
            safe_mask[done] = False
            safe_mask[done, 0] = True

            probs = safe_mask.float() / safe_mask.float().sum(dim=-1, keepdim=True)
            probs_flat = probs.view(-1, probs.shape[-1])
            action_flat = torch.multinomial(probs_flat, 1).squeeze(-1)

            action = action_flat.view(*probs.shape[:-1])
            actions_taken.append(action)

            state["action"] = action
            state = env.step(state)["next"]

            done = done | state["done"]
            if done.all():
                break

        actions_tensor = torch.stack(actions_taken, dim=1)
        reward = env.get_reward(state, actions_tensor)
        cost = state["cost"]

    # --- Print results per batch instance ---
    print("\n" + "=" * 60)
    for b in range(batch_size):
        actions_list = actions_tensor[b].tolist()

        # Split action sequence into per-vehicle routes at depot visits
        routes = []
        current_route = [0]
        for a in actions_list:
            if a == 0 and len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
            elif a != 0:
                current_route.append(a)
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)

        n_vehicles = len(routes)
        excess = max(0, n_vehicles - env.free_vehicles)
        veh_pen = env.vehicle_penalty * excess

        print(f"\nBatch {b}:")
        print(f"  Vehicles used: {n_vehicles}")
        for v_idx, route in enumerate(routes):
            print(f"  Vehicle {v_idx + 1}: {route}")
        print(f"  Routing cost:     {cost[b].item():.2f}")
        print(f"  Vehicle penalty:  {veh_pen:.2f}  ({excess} excess x {env.vehicle_penalty})")
        print(f"  Total reward:     {reward[b].item():.2f}")
    print("=" * 60)
