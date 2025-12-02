from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Unbounded, Bounded, Composite

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.cvrptw.env import CVRPTWEnv
from rl4co.utils.ops import gather_by_index
from .generator import PDPTWGenerator
from .sf_generator import SFGenerator
from .render import render

class PDPTWEnv(CVRPTWEnv):
    """
    Pickup and Delivery Problem with Time Windows (PDPTW) environment.
    
    Logic:
    - Depot returns are restricted: A vehicle must deliver all items on board before returning,
      unless no other moves are possible (fallback).
    - Masking: 1-step Lookahead. Ensures that visiting a node does not make any current 
      obligation immediately impossible to fulfill.
    """
    name = "pdptw"

    def __init__(
        self,
        generator: SFGenerator = None,
        generator_params: dict = {},
        depot_penalty: int = 100,
        fleet_count: int = 5,
        unresolved_penalty: int = 200,
        **kwargs,
    ):
        self.depot_penalty_factor = int(depot_penalty)
        self.fleet_count = fleet_count
        self.unresolved_penalty = unresolved_penalty
        
        if generator is None:
            generator = SFGenerator(**generator_params)
        super().__init__(generator=generator, **kwargs)
        self.generator = generator
        self._make_spec(self.generator)

    def _make_spec(self, generator):
        if not hasattr(generator, 'num_customers'):
            return

        self.observation_spec = Composite(
            h3_indices=Bounded(low=0, high=generator.travel_time_matrix.shape[0], shape=(generator.num_customers * 2 + 1,), dtype=torch.int64),
            time_windows=Bounded(low=0, high=1440, shape=(generator.num_customers * 2 + 1, 2), dtype=torch.float32),
            demand=Bounded(low=-generator.vehicle_capacity, high=generator.vehicle_capacity, shape=(generator.num_customers * 2 + 1,), dtype=torch.float32),
            current_node=Unbounded(shape=(1), dtype=torch.int64),
            current_time=Bounded(low=0, high=1440, shape=(1), dtype=torch.float32),
            used_capacity=Bounded(low=0, high=generator.vehicle_capacity, shape=(1), dtype=torch.float32),
            pending_schedule=Bounded(low=0, high=generator.num_customers * 2, shape=(generator.vehicle_capacity,), dtype=torch.int64),
            pending_count=Bounded(low=0, high=generator.vehicle_capacity, shape=(1), dtype=torch.int64),
            visited=Bounded(low=0, high=1, shape=(generator.num_customers * 2 + 1,), dtype=torch.bool),
            action_mask=Bounded(low=0, high=1, shape=(generator.num_customers * 2 + 1,), dtype=torch.bool),
            previous_action=Unbounded(shape=(1), dtype=torch.int64),
            i=Unbounded(shape=(1), dtype=torch.int64),
            flexibility=Bounded(low=-30, high=30, shape=(generator.num_customers * 2 + 1,), dtype=torch.float32)
        )
        self.action_spec = Bounded(
            shape=(1,), dtype=torch.int64, low=0, high=generator.num_customers * 2 + 1
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        
        h3_indices = td["h3_indices"]
        demands = td["demand"]
        tws = td["time_windows"]
        travel_time_matrix = td["travel_time_matrix"]
        locs = td.get("locs", None)
        flexibility = td.get("flexibility", torch.zeros_like(demands))

        capacity_limit = self.generator.vehicle_capacity
        pending_schedule = torch.zeros((*batch_size, capacity_limit), dtype=torch.int64, device=device)
        pending_count = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        td_reset = TensorDict({
            "h3_indices": h3_indices,
            "travel_time_matrix": travel_time_matrix,
            "time_windows": tws,
            "demand": demands,
            "flexibility": flexibility,
            "current_node": torch.zeros(*batch_size, 1, dtype=torch.int64, device=device),
            "current_time": torch.zeros(*batch_size, 1, dtype=torch.float32, device=device),
            "used_capacity": torch.zeros(*batch_size, 1, dtype=torch.float32, device=device),
            "visited": torch.zeros(*batch_size, h3_indices.shape[-1], dtype=torch.bool, device=device),
            "vehicle_capacity": torch.full((*batch_size, 1), capacity_limit, device=device),
            "pending_schedule": pending_schedule,
            "pending_count": pending_count,
            "previous_action": torch.zeros(*batch_size, 1, dtype=torch.int64, device=device),
            "i": torch.zeros(*batch_size, 1, dtype=torch.int64, device=device),
        }, batch_size=batch_size)

        if locs is not None:
            td_reset.set("locs", locs)

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _get_travel_time(self, td, from_node_idx, to_node_idx):
        from_h3 = gather_by_index(td["h3_indices"], from_node_idx, squeeze=False)
        to_h3 = gather_by_index(td["h3_indices"], to_node_idx, squeeze=False)
        rows = gather_by_index(td["travel_time_matrix"], from_h3, dim=1, squeeze=False) 
        n_from = from_h3.size(1)
        n_to = to_h3.size(1)
        if n_from == 1 and n_to > 1:
            rows = rows.expand(-1, n_to, -1)
        elif n_from > 1 and n_to == 1:
            to_h3 = to_h3.expand(-1, n_from)
        return gather_by_index(rows, to_h3, dim=2)

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]
        batch_size = action.shape[0]

        curr_node = td["current_node"]
        
        # Calculate arrival
        travel_time = self._get_travel_time(td, curr_node, action.unsqueeze(-1))
        arrival_time = td["current_time"] + travel_time
        start_window = gather_by_index(td["time_windows"], action)[..., 0]
        service_start_time = torch.max(arrival_time, start_window.unsqueeze(-1))

        # Reset load/time if returning to depot
        is_return_to_depot = (action == 0) & (curr_node.squeeze(-1) != 0)
        service_start_time = torch.where(is_return_to_depot.unsqueeze(-1), torch.zeros_like(service_start_time), service_start_time)
        
        # Capacity updates
        selected_demand = gather_by_index(td["demand"], action).unsqueeze(-1)
        new_load = td["used_capacity"] + selected_demand
        new_load = torch.where(is_return_to_depot.unsqueeze(-1), torch.zeros_like(new_load), new_load)

        # Visited updates
        new_visited = td["visited"].clone()
        non_depot_mask = (action != 0).unsqueeze(-1)
        new_visited.scatter_(-1, action.unsqueeze(-1), non_depot_mask)

        # --- Graph / Schedule Update ---
        pending_schedule = td["pending_schedule"].clone()
        pending_count = td["pending_count"].clone()
        is_pickup = (action % 2 != 0) & (action != 0)
        
        # 1. Remove visited node from schedule
        mask_in_schedule = (pending_schedule == action.unsqueeze(-1))
        pending_schedule[mask_in_schedule] = 0
        
        # 2. Update count based on dropoff
        is_dropoff = (action % 2 == 0) & (action != 0)
        pending_count = torch.clamp(pending_count - is_dropoff.long().unsqueeze(-1), min=0)

        # 3. Insert Partner for new pickups
        num_nodes = td["h3_indices"].shape[1]
        partner_node = torch.clamp(action + 1, max=num_nodes - 1)
        new_entry = torch.zeros((batch_size, 1), dtype=torch.int64, device=td.device)
        new_entry[is_pickup, 0] = partner_node[is_pickup]
        combined = torch.cat([pending_schedule, new_entry], dim=1)
        
        # 4. Compact schedule (Move zeros to the end)
        # We don't strictly need start-time sorting for the "Star Check", 
        # so compacting is sufficient and faster.
        node_vals = combined.clone()
        node_vals[combined == 0] = 1e9 # Push zeros to end
        _, sort_idx = torch.sort(node_vals, dim=1)
        sorted_sched = torch.gather(combined, 1, sort_idx)
        
        capacity = td["vehicle_capacity"].shape[1] if len(td["vehicle_capacity"].shape)>1 else int(td["vehicle_capacity"][0].item())
        pending_schedule = sorted_sched[:, :capacity]
        
        pending_count = torch.clamp(pending_count + is_pickup.long().unsqueeze(-1), max=capacity)
        
        # Clear schedule if returned to depot (Items dropped/failed)
        pending_schedule = torch.where(is_return_to_depot.unsqueeze(-1), torch.zeros_like(pending_schedule), pending_schedule)
        pending_count = torch.where(is_return_to_depot.unsqueeze(-1), torch.zeros_like(pending_count), pending_count)

        td_out = td.clone()
        td_out.update({
            "current_node": action.unsqueeze(-1),
            "current_time": service_start_time,
            "used_capacity": new_load,
            "visited": new_visited,
            "pending_schedule": pending_schedule,
            "pending_count": pending_count,
            "previous_action": action.unsqueeze(-1),
            "i": td["i"] + 1
        })

        all_visited = new_visited[..., 1:].all()
        is_depot = (action == 0)
        standard_done = is_depot & all_visited
        previous_was_depot = (td["previous_action"].squeeze(-1) == 0)
        double_depot = is_depot & previous_was_depot
        
        td_out.set("done", standard_done | double_depot)
        td_out.set("action_mask", self.get_action_mask(td_out))
        return td_out

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        batch_size = td["h3_indices"].shape[0]
        num_nodes = td["h3_indices"].shape[1]
        device = td["h3_indices"].device
        capacity = td["pending_schedule"].shape[1]

        # =========================================================================
        # 0. Base Logic Masks (Structure, Precedence, Capacity, Depot)
        # =========================================================================
        visited = td["visited"]
        current_node = td["current_node"].squeeze(-1)
        current_time = td["current_time"]
        
        node_indices = torch.arange(num_nodes, device=device).expand(batch_size, num_nodes)
        
        is_depot = (node_indices == 0)
        is_pickup = (node_indices % 2 != 0) & (~is_depot)
        is_dropoff = (node_indices % 2 == 0) & (~is_depot)

        # Precedence Constraints
        pickup_mask = ~visited & is_pickup
        partner_indices = torch.clamp(node_indices - 1, min=0)
        partner_visited = torch.gather(visited, 1, partner_indices)
        dropoff_mask = partner_visited & (~visited) & is_dropoff
        
        # Depot Constraints: Return only if empty OR all visited (Prison Logic)
        all_visited = visited[..., 1:].all(dim=-1, keepdim=True)
        is_empty = (td["pending_count"] == 0)
        can_return_to_depot = is_empty & (current_node.unsqueeze(-1) != 0)
        depot_mask = is_depot & (all_visited | can_return_to_depot)

        precedence_mask = pickup_mask | dropoff_mask | depot_mask
        
        # Capacity Constraints
        remaining_cap = td["vehicle_capacity"] - td["used_capacity"]
        demand_ok = (td["demand"] <= remaining_cap) | is_dropoff | is_depot
        
        base_mask = precedence_mask & demand_ok

        # =========================================================================
        # 1. Level 1: Immediate Reachability
        #    Can we reach Candidate from Current within time window?
        # =========================================================================
        travel_time_to_candidate = self._get_travel_time(td, current_node.unsqueeze(-1), node_indices)
        arrival_at_candidate = current_time + travel_time_to_candidate
        
        candidate_late = td["time_windows"][..., 1]
        candidate_early = td["time_windows"][..., 0]
        
        # Note: arrival <= late_window
        level1_mask = (arrival_at_candidate <= candidate_late)
        
        current_mask = base_mask & level1_mask

        # =========================================================================
        # 2. Level 2: One-Step Star Lookahead
        #    Check: Current -> Candidate -> Undelivered[I] (for all I)
        #    If going to Candidate makes reaching ANY pending node impossible, forbid it.
        # =========================================================================
        
        # Simulation state: Arriving at Candidate
        # Departure/Availability time at Candidate = max(arrival, start)
        sim_time_at_candidate = torch.max(arrival_at_candidate, candidate_early) # (Batch, N)
        
        # We iterate over all pending slots (Capacity)
        # and invalidate candidates that fail to reach any valid target.
        star_valid = torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)
        
        for k in range(capacity):
            # Get target node for this slot
            target = td["pending_schedule"][:, k].unsqueeze(-1) # (Batch, 1)
            target_expanded = target.expand(batch_size, num_nodes) # (Batch, N)
            
            # Condition 1: Slot must be active (not 0)
            has_target = (target_expanded != 0)
            
            # Condition 2: Target must NOT be the candidate itself
            # (If we are visiting '8' now, we don't check if we can reach '8' later)
            is_satisfied = (target_expanded == node_indices)
            
            check_needed = has_target & (~is_satisfied)
            
            # Calculate Travel: Candidate -> Target
            # (Batch, N) -> (Batch, N)
            dt = self._get_travel_time(td, node_indices, target_expanded)
            arr_at_target = sim_time_at_candidate + dt
            
            # Target Constraints
            t_late = gather_by_index(td["time_windows"], target_expanded)[..., 1]
            
            # Feasibility Check
            is_late = (arr_at_target > t_late)
            
            # If we need to reach this target, and we are late -> Candidate is invalid
            # Logic: star_valid becomes False if (CheckNeeded AND IsLate)
            star_valid = star_valid & ~(check_needed & is_late)

        final_mask = current_mask & star_valid

        # =========================================================================
        # 3. Fallbacks (Safety)
        # =========================================================================
        # If no feasible nodes exist, allow depot if empty (wait logic/end logic)
        no_feasible = ~final_mask[..., 1:].any(dim=-1, keepdim=True)
        final_mask[..., 0] = final_mask[..., 0] | (no_feasible & is_empty).squeeze(-1)
        
        # Absolute fallback: If truly nothing allowed (deadlock with items on board),
        # allow depot to dump items and prevent runtime crash.
        none_at_all = ~final_mask.any(dim=-1, keepdim=True)
        final_mask[..., 0] = final_mask[..., 0] | none_at_all.squeeze(-1)

        return final_mask

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        ordered_nodes = torch.cat([torch.zeros_like(actions[:, 0:1]), actions], dim=1)
        travel_times = self._get_travel_time(td, ordered_nodes[:, :-1], ordered_nodes[:, 1:])
        distance_reward = -travel_times.sum(dim=1)
        td.set("cost", travel_times.sum(dim=1))

        batch_size = actions.shape[0]
        device = actions.device
        num_nodes = td["h3_indices"].shape[1]

        on_board = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)
        total_unresolved = torch.zeros(batch_size, dtype=torch.float32, device=device)

        for t in range(actions.shape[1]):
            node = actions[:, t]
            is_depot = (node == 0)
            is_pickup = (node % 2 != 0) & (~is_depot)
            is_dropoff = (node % 2 == 0) & (~is_depot)

            penalties_at_step = on_board.sum(dim=1).float()
            total_unresolved = torch.where(is_depot, total_unresolved + penalties_at_step, total_unresolved)
            on_board = torch.where(is_depot.unsqueeze(-1), torch.zeros_like(on_board), on_board)

            on_board.scatter_(1, node.unsqueeze(-1), is_pickup.unsqueeze(-1))
            partner_node = torch.clamp(node - 1, min=0)
            remove_mask = torch.zeros_like(on_board)
            remove_mask.scatter_(1, partner_node.unsqueeze(-1), is_dropoff.unsqueeze(-1))
            on_board = on_board & (~remove_mask)

        total_unresolved += on_board.sum(dim=1).float()
        unresolved_penalty = total_unresolved * self.unresolved_penalty
        td.set("unresolved_penalty", unresolved_penalty)
        return distance_reward - unresolved_penalty

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        pass 

    @staticmethod
    def render(td, actions=None, ax=None):
        return render(td, actions, ax)