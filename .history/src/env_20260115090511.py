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
    
    Masking Strategy:
    - Immediate reachability: arrival must be within the candidate's time window
    - One-step lookahead: after visiting a candidate, all onboard dropoffs 
      (including the new pickup if applicable) must still be reachable within their time windows
    Dynamic Visitation Management:
    - If a route segment returns to depot with pickups whose deliveries were not 
      completed, those pickups are omitted from the effective visitation history
    - Travel reward is computed on the effective route (omitting unresolved pickups)
    - Returning to depot starts a new vehicle with time reset to zero; vehicles used are counted
    """
    
    def __init__(
        self,
        generator: Optional[Callable] = None,
        vehicle_capacity: Optional[int] = None,
        vehicle_penalty: float = -20.0,
        **kwargs
    ):
        super().__init__()
        self.generator = generator
        if vehicle_capacity is None and hasattr(generator, "vehicle_capacity"):
            vehicle_capacity = getattr(generator, "vehicle_capacity")
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_penalty = float(vehicle_penalty)
        
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
        
        device = td.device if td.device is not None else td["h3_indices"].device
        
        # Extract problem data
        h3_indices = td["h3_indices"]
        demands = td["demand"]
        time_windows = td["time_windows"]
        travel_time_matrix = td["travel_time_matrix"]
        locs = td.get("locs", None)
        flexibility = td.get("flexibility", torch.zeros_like(demands))
        
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
        device = td.device if td.device is not None else td["h3_indices"].device
        
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
        
        # Build output state
        td_out = td.clone()
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
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward for completed episode.
        
        Args:
            td: TensorDict with problem data
            actions: Sequence of actions taken [batch_size, seq_len]
            
        Returns:
            Reward tensor [batch_size]
        """
        effective_routes = self._build_effective_routes(actions)
        
        costs = []
        for batch_idx, route in enumerate(effective_routes):
            cost = self._travel_time_for_route(td, batch_idx, route)
            costs.append(cost)
        cost_tensor = torch.stack(costs)
        td.set("cost", cost_tensor)

        penalty = None
        if "vehicles_used" in td.keys():
            vehicles_used = td["vehicles_used"].squeeze(-1).float()
            penalty = vehicles_used * self.vehicle_penalty
            td.set("vehicle_penalty", penalty)

        reward = -cost_tensor
        if penalty is not None:
            reward = reward + penalty
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
    
    def _compute_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask using immediate reachability and one-step lookahead.
        
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
        
        # Base availability
        can_pickup = is_pickup & ~td["completed"] & ~onboard_pickups
        remaining_cap = td["vehicle_capacity"] - td["used_capacity"]
        demand_ok = (td["demand"] <= remaining_cap) | ~is_pickup
        can_pickup = can_pickup & demand_ok
        
        pickup_indices = torch.clamp(node_indices - 1, min=0)
        pickup_visited = td["visited"].gather(1, pickup_indices)
        can_dropoff = is_dropoff & pickup_visited & ~td["completed"]
        
        current_not_depot = (td["current_node"].squeeze(-1) != 0).unsqueeze(-1)
        base_mask = (is_depot & current_not_depot) | can_pickup | can_dropoff
        
        # Immediate reachability
        travel_time_to_candidate = self._get_travel_time(
            td, 
            current_node.unsqueeze(-1), 
            node_indices
        )
        arrival_at_candidate = current_time + travel_time_to_candidate
        candidate_early = td["time_windows"][..., 0]
        candidate_late = td["time_windows"][..., 1]
        time_at_candidate = torch.max(arrival_at_candidate, candidate_early)
        
        time_feasible = arrival_at_candidate <= candidate_late
        current_mask = base_mask & time_feasible
        
        lookahead_valid = self._lookahead_validity(
            td,
            time_at_candidate,
            node_indices,
            num_nodes
        )

        mask = current_mask & (~is_pickup | lookahead_valid)

        # Mask depot if any dropoff is feasible; otherwise allow depot.
        reachable_dropoff = (mask & is_dropoff).any(dim=-1, keepdim=True)
        allow_depot = (~reachable_dropoff & current_not_depot).squeeze(-1)
        mask[..., 0] = allow_depot
        
        return mask
    
    def _lookahead_validity(
        self,
        td: TensorDict,
        time_at_candidate: torch.Tensor,
        node_indices: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        One-step lookahead: ensure every onboard dropoff (plus the new pickup's
        dropoff, if any) can still be reached in time after choosing a candidate.
        """
        batch_size = node_indices.shape[0]
        device = node_indices.device
        capacity = td["pending_schedule"].shape[1]
        
        star_valid = torch.ones_like(time_at_candidate, dtype=torch.bool)
        
        # Existing onboard obligations
        for k in range(capacity):
            target = td["pending_schedule"][:, k].unsqueeze(-1)
            target_expanded = target.expand(batch_size, num_nodes)
            
            has_target = target_expanded != 0
            travel_to_target = self._get_travel_time(td, node_indices, target_expanded)
            arrival_at_target = time_at_candidate + travel_to_target
            target_late = gather_by_index(td["time_windows"], target_expanded)[..., 1]
            
            is_late = arrival_at_target > target_late
            star_valid = star_valid & ~(has_target & is_late)
        
        # Potential new dropoff if the candidate is a pickup
        candidate_dropoff = torch.where(
            (node_indices % 2 != 0) & (node_indices != 0),
            torch.clamp(node_indices + 1, max=num_nodes - 1),
            torch.zeros_like(node_indices)
        )
        has_new_drop = candidate_dropoff != 0
        travel_new = self._get_travel_time(td, node_indices, candidate_dropoff)
        arrival_new = time_at_candidate + travel_new
        drop_late = gather_by_index(td["time_windows"], candidate_dropoff)[..., 1]
        star_valid = star_valid & ~(has_new_drop & (arrival_new > drop_late))
        
        return star_valid
    
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

    def _build_effective_routes(self, actions: torch.Tensor) -> list[list[int]]:
        """
        Remove pickups whose deliveries were not completed before returning to depot.
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
        if len(route) < 2:
            return torch.tensor(0.0, device=td.device)
        
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
    # Example using SFGenerator for data and random actions
    from generator import SFGenerator

    batch_size = 3
    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")

    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path
    )

    env = PDPTWEnv(generator=generator)

    state = env.reset(batch_size=[batch_size])
    print(f"Initial action mask counts: {state['action_mask'].sum(dim=-1)}")

    actions_taken = []
    max_steps = 50
    for step in range(max_steps):
        mask = state["action_mask"]
        probs = mask.float() / mask.float().sum(dim=-1, keepdim=True)
        probs_flat = probs.view(-1, probs.shape[-1])
        try: 
            action_flat = torch.multinomial(probs_flat, 1).squeeze(-1)
        except RuntimeError as e:
            raise RuntimeError("At least one environment has finished - invalid probability distribution") from e

        action = action_flat.view(*probs.shape[:-1])
        actions_taken.append(action)

        state["action"] = action
        state = env.step(state)["next"]

        print(
            f"Step {step+1}: action={action.tolist()}, "
            f"done={state['done'].tolist()}, "
            f"vehicles_used={state['vehicles_used'].squeeze(-1).tolist()}, "
            f"valid_next={state['action_mask'].sum(dim=-1).tolist()}"
        )

        if state["done"].all():
            break

    actions_tensor = torch.stack(actions_taken, dim=1)
    reward = env.get_reward(state, actions_tensor)
    print(f"Episode reward for given steps: {reward.tolist()}")
