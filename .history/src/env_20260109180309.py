from __future__ import annotations
from typing import Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict


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
        expanded_index = index.unsqueeze(-1).expand(*index.shape, source.shape[-1])
        result = source.gather(dim, expanded_index)
    elif dim == 2:
        result = source.gather(dim, index)
    else:
        expanded_index = index.expand(*source.shape[:-1], index.shape[-1])
        result = source.gather(dim, expanded_index)
    
    if squeeze and result.shape[-1] == 1:
        result = result.squeeze(-1)
    return result


@dataclass
class PDPTWConfig:
    """Configuration for PDPTW environment."""
    num_customers: int = 50
    vehicle_capacity: int = 10
    fleet_count: int = 5
    depot_penalty: float = 0.0
    unresolved_penalty: float = 0.0
    max_time: float = 1440.0  # Minutes in a day


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
    - Base constraints: visited, precedence, capacity
    - Lookahead: For all visited but not dropped off nodes in the schedule, 
      Visiting a new node of interest must be such that time constraints allow all to-be-dropped off nodes still to be visited.
      (In other words, if there is a visitation 1,3,5,4,2; then 6 is to-be-dropped off. 
      Visiting node 7 must still be able to satisfy time window for either 6 or 8; 
      if we are at time t, then max ( t + travel_time(2,7), time_window(7)[0] ) + travel_time(7,6) < time_window(6)[1] )
    * The following are new requirements to incorporate into the PDPTW Env
    Dynamic Visitation Management: 
    - Problem Context: One step lookahead doesn't guarantee the 
        existence of a Valid Traveling-Salesmen-Problem with Time Window Solution for undelivered nodes. 
    - Solution: implement the following logic in node selection: 
        - Policy constructs a route according to the current masking scheme (Immediate Reachability and One-step Lookahead)
        - There might be some violations (defined as nodes picked up but not dropped off): 
          For example, route 0,3,5,7,8,9,10,0 has violations because node 4 and 6 are not visited. 
        - Instead of penalizing the violations, we omit the node 3 and 5 form the current visitation history; 
        - The actual route taken is then the current route without node 3 and 5; and the reward must be calculated as such (node connection without 3 and 5)
        - Start anew; returning to 0 = a new vehicle and vehicle time is dialed back to 0. 
        - Keep track of how many vehicles are used; but there is not an upper limit for vehicles. 
    """
    
    def __init__(
        self,
        config: Optional[PDPTWConfig] = None,
        generator: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__()
        self.config = config or PDPTWConfig(**kwargs)
        self.generator = generator
        
        # Extract config values for convenience
        self.num_customers = self.config.num_customers
        self.vehicle_capacity = self.config.vehicle_capacity
        self.fleet_count = self.config.fleet_count
        self.depot_penalty = self.config.depot_penalty
        self.unresolved_penalty = self.config.unresolved_penalty
        
        # Total nodes = depot + 2*customers (pickup + delivery for each)
        self.num_nodes = self.num_customers * 2 + 1
        
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
        
        device = td.device
        
        # Extract problem data
        h3_indices = td["h3_indices"]
        demands = td["demand"]
        time_windows = td["time_windows"]
        travel_time_matrix = td["travel_time_matrix"]
        locs = td.get("locs", None)
        flexibility = td.get("flexibility", torch.zeros_like(demands))
        
        # Initialize state
        capacity_limit = self.vehicle_capacity
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
            "vehicle_capacity": torch.full(
                (*batch_size, 1), 
                capacity_limit, 
                device=device
            ),
            
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
        device = td.device
        
        curr_node = td["current_node"]
        
        # Calculate arrival time at selected node
        travel_time = self._get_travel_time(td, curr_node, action.unsqueeze(-1))
        arrival_time = td["current_time"] + travel_time
        
        # Service starts at max(arrival, window_start)
        start_window = gather_by_index(td["time_windows"], action)[..., 0]
        service_start_time = torch.max(arrival_time, start_window.unsqueeze(-1))
        
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
        
        # Update visited nodes (don't mark depot as visited)
        new_visited = td["visited"].clone()
        non_depot_mask = (action != 0).unsqueeze(-1)
        new_visited.scatter_(-1, action.unsqueeze(-1), non_depot_mask)
        
        # Update pending schedule (obligations)
        pending_schedule, pending_count = self._update_pending_schedule(
            td, action, is_return_to_depot, batch_size, device
        )
        
        # Build output state
        td_out = td.clone()
        td_out.update({
            "current_node": action.unsqueeze(-1),
            "current_time": service_start_time,
            "used_capacity": new_load,
            "visited": new_visited,
            "pending_schedule": pending_schedule,
            "pending_count": pending_count,
            "previous_action": action.unsqueeze(-1),
            "i": td["i"] + 1,
        })
        
        # Check termination conditions
        all_visited = new_visited[..., 1:].all(dim=-1)
        is_depot = (action == 0)
        standard_done = is_depot & all_visited
        previous_was_depot = (td["previous_action"].squeeze(-1) == 0)
        double_depot = is_depot & previous_was_depot
        
        td_out.set("done", standard_done | double_depot)
        td_out.set("action_mask", self._compute_action_mask(td_out))
        
        return td_out
    
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
        # Compute travel time (negative because we minimize)
        ordered_nodes = torch.cat(
            [torch.zeros_like(actions[:, 0:1]), actions], 
            dim=1
        )
        travel_times = self._get_travel_time(
            td, 
            ordered_nodes[:, :-1], 
            ordered_nodes[:, 1:]
        )
        distance_reward = -travel_times.sum(dim=1)
        td.set("cost", travel_times.sum(dim=1))
        
        # Compute penalty for unresolved pickups
        unresolved_penalty = self._compute_unresolved_penalty(
            td, actions
        )
        td.set("unresolved_penalty", unresolved_penalty)
        
        return distance_reward - unresolved_penalty
    
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
        # Map node indices to H3 indices (location indices in matrix)
        from_h3 = gather_by_index(td["h3_indices"], from_node_idx, squeeze=False)
        to_h3 = gather_by_index(td["h3_indices"], to_node_idx, squeeze=False)
        
        # Gather rows from travel time matrix
        rows = gather_by_index(
            td["travel_time_matrix"], 
            from_h3, 
            dim=1, 
            squeeze=False
        )
        
        # Handle broadcasting for different from/to sizes
        n_from = from_h3.size(1)
        n_to = to_h3.size(1)
        if n_from == 1 and n_to > 1:
            rows = rows.expand(-1, n_to, -1)
        elif n_from > 1 and n_to == 1:
            to_h3 = to_h3.expand(-1, n_from)
        
        return gather_by_index(rows, to_h3, dim=2)
    
    # =========================================================================
    # Action Masking
    # =========================================================================
    
    def _compute_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute action mask using multi-level constraints.
        
        Level 0: Base constraints (structure, precedence, capacity, depot)
        Level 1: Immediate reachability (time window feasibility)
        Level 2: One-step lookahead (star check for pending obligations)
        
        Args:
            td: Current state TensorDict
            
        Returns:
            Boolean mask [batch_size, num_nodes] - True if action is valid
        """
        batch_size = td["h3_indices"].shape[0]
        num_nodes = td["h3_indices"].shape[1]
        device = td["h3_indices"].device
        capacity = td["pending_schedule"].shape[1]
        
        # Level 0: Base constraints
        base_mask = self._compute_base_mask(td, batch_size, num_nodes, device)
        
        # Level 1: Immediate reachability
        current_node = td["current_node"].squeeze(-1)
        current_time = td["current_time"]
        node_indices = torch.arange(num_nodes, device=device).expand(batch_size, -1)
        
        travel_time_to_candidate = self._get_travel_time(
            td, 
            current_node.unsqueeze(-1), 
            node_indices
        )
        arrival_at_candidate = current_time + travel_time_to_candidate
        candidate_late = td["time_windows"][..., 1]
        
        level1_mask = arrival_at_candidate <= candidate_late
        current_mask = base_mask & level1_mask
        
        # Level 2: One-step lookahead (star check)
        star_valid = self._compute_star_validity(
            td, 
            arrival_at_candidate,
            node_indices,
            batch_size, 
            num_nodes, 
            capacity, 
            device
        )
        
        final_mask = current_mask & star_valid
        
        # Fallback safety: ensure at least depot is available
        final_mask = self._apply_fallbacks(
            final_mask, 
            td["pending_count"] == 0,
            device
        )
        
        return final_mask
    
    def _compute_base_mask(
        self,
        td: TensorDict,
        batch_size: int,
        num_nodes: int,
        device: torch.device
    ) -> torch.Tensor:
        """Compute base feasibility mask (precedence, capacity, depot rules)."""
        visited = td["visited"]
        current_node = td["current_node"].squeeze(-1)
        
        node_indices = torch.arange(num_nodes, device=device).expand(batch_size, -1)
        
        # Node type classification
        is_depot = node_indices == 0
        is_pickup = (node_indices % 2 != 0) & ~is_depot
        is_dropoff = (node_indices % 2 == 0) & ~is_depot
        
        # Precedence: can visit unvisited pickups
        pickup_mask = ~visited & is_pickup
        
        # Precedence: can visit dropoff only if pickup was visited
        partner_indices = torch.clamp(node_indices - 1, min=0)
        partner_visited = torch.gather(visited, 1, partner_indices)
        dropoff_mask = partner_visited & ~visited & is_dropoff
        
        # Depot: can return if empty or all visited
        all_visited = visited[..., 1:].all(dim=-1, keepdim=True)
        is_empty = td["pending_count"] == 0
        can_return = is_empty & (current_node.unsqueeze(-1) != 0)
        depot_mask = is_depot & (all_visited | can_return)
        
        precedence_mask = pickup_mask | dropoff_mask | depot_mask
        
        # Capacity: can visit if demand fits
        remaining_cap = td["vehicle_capacity"] - td["used_capacity"]
        demand_ok = (td["demand"] <= remaining_cap) | is_dropoff | is_depot
        
        return precedence_mask & demand_ok
    
    def _compute_star_validity(
        self,
        td: TensorDict,
        arrival_at_candidate: torch.Tensor,
        node_indices: torch.Tensor,
        batch_size: int,
        num_nodes: int,
        capacity: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        One-step lookahead: check if visiting each candidate allows 
        reaching all pending obligations.
        
        For each candidate C and each pending target T:
        - Simulate arriving at C
        - Check if we can still reach T before its deadline
        """
        candidate_early = td["time_windows"][..., 0]
        sim_time_at_candidate = torch.max(arrival_at_candidate, candidate_early)
        
        star_valid = torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)
        
        for k in range(capacity):
            target = td["pending_schedule"][:, k].unsqueeze(-1)
            target_expanded = target.expand(batch_size, num_nodes)
            
            # Only check active targets that aren't the candidate itself
            has_target = target_expanded != 0
            is_satisfied = target_expanded == node_indices
            check_needed = has_target & ~is_satisfied
            
            # Compute travel time: candidate -> target
            travel_to_target = self._get_travel_time(td, node_indices, target_expanded)
            arrival_at_target = sim_time_at_candidate + travel_to_target
            
            # Get target's late window
            target_late = gather_by_index(td["time_windows"], target_expanded)[..., 1]
            
            # Mark invalid if we'd be late
            is_late = arrival_at_target > target_late
            star_valid = star_valid & ~(check_needed & is_late)
        
        return star_valid
    
    def _apply_fallbacks(
        self,
        mask: torch.Tensor,
        is_empty: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Apply fallback rules to ensure at least one valid action."""
        # If no customer nodes feasible, allow depot return if empty
        no_feasible = ~mask[..., 1:].any(dim=-1, keepdim=True)
        mask[..., 0] = mask[..., 0] | (no_feasible & is_empty).squeeze(-1)
        
        # Absolute fallback: if nothing valid, force depot
        none_at_all = ~mask.any(dim=-1, keepdim=True)
        mask[..., 0] = mask[..., 0] | none_at_all.squeeze(-1)
        
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
    # Penalty Computation
    # =========================================================================
    
    def _compute_unresolved_penalty(
        self,
        td: TensorDict,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute penalty for items returned to depot without delivery."""
        batch_size = actions.shape[0]
        device = actions.device
        num_nodes = td["h3_indices"].shape[1]
        
        on_board = torch.zeros(
            (batch_size, num_nodes), 
            dtype=torch.bool, 
            device=device
        )
        total_unresolved = torch.zeros(
            batch_size, 
            dtype=torch.float32, 
            device=device
        )
        
        for t in range(actions.shape[1]):
            node = actions[:, t]
            is_depot = node == 0
            is_pickup = (node % 2 != 0) & ~is_depot
            is_dropoff = (node % 2 == 0) & ~is_depot
            
            # Penalize items on board when returning to depot
            penalties_at_step = on_board.sum(dim=1).float()
            total_unresolved = torch.where(
                is_depot,
                total_unresolved + penalties_at_step,
                total_unresolved
            )
            
            # Clear on depot return
            on_board = torch.where(
                is_depot.unsqueeze(-1),
                torch.zeros_like(on_board),
                on_board
            )
            
            # Add pickup to on_board
            on_board.scatter_(1, node.unsqueeze(-1), is_pickup.unsqueeze(-1))
            
            # Remove delivered item
            partner_node = torch.clamp(node - 1, min=0)
            remove_mask = torch.zeros_like(on_board)
            remove_mask.scatter_(1, partner_node.unsqueeze(-1), is_dropoff.unsqueeze(-1))
            on_board = on_board & ~remove_mask
        
        # Final unresolved items
        total_unresolved += on_board.sum(dim=1).float()
        
        return total_unresolved * self.unresolved_penalty


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example with mock data
    batch_size = [4]
    num_customers = 10
    num_nodes = num_customers * 2 + 1
    num_h3_cells = 50
    
    # Create mock problem instance
    td = TensorDict({
        "h3_indices": torch.randint(0, num_h3_cells, (*batch_size, num_nodes)),
        "demand": torch.cat([
            torch.zeros(*batch_size, 1),  # Depot
            torch.ones(*batch_size, num_customers),  # Pickups (+1)
            -torch.ones(*batch_size, num_customers),  # Deliveries (-1)
        ], dim=-1),
        "time_windows": torch.stack([
            torch.zeros(*batch_size, num_nodes),  # Early
            torch.full((*batch_size, num_nodes), 1440.0),  # Late
        ], dim=-1),
        "travel_time_matrix": torch.rand(*batch_size, num_h3_cells, num_h3_cells) * 30,
    }, batch_size=batch_size)
    
    # Create environment
    config = PDPTWConfig(
        num_customers=num_customers,
        vehicle_capacity=5,
        unresolved_penalty=100.0
    )
    env = PDPTWEnv(config=config)
    
    # Reset
    state = env.reset(td=td, batch_size=batch_size)
    print(f"Initial state shape: {state.batch_size}")
    print(f"Action mask shape: {state['action_mask'].shape}")
    print(f"Valid actions at start: {state['action_mask'].sum(dim=-1)}")
    
    # Take a few random valid steps
    actions_taken = []
    for step in range(5):
        # Select random valid action
        mask = state["action_mask"]
        valid_actions = mask.float()
        probs = valid_actions / valid_actions.sum(dim=-1, keepdim=True)
        action = torch.multinomial(probs, 1).squeeze(-1)
        actions_taken.append(action)
        
        state["action"] = action
        state = env.step(state)
        
        print(f"Step {step+1}: action={action.tolist()}, "
              f"done={state['done'].tolist()}, "
              f"valid_next={state['action_mask'].sum(dim=-1).tolist()}")
        
        if state["done"].all():
            break
    
    print("\nEnvironment test completed successfully!")