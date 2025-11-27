from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Unbounded, Bounded, Composite

from inner_loop.rl4co.envs.routing.cvrp.env import CVRPEnv
from inner_loop.rl4co.envs.routing.cvrptw.env import CVRPTWEnv
from inner_loop.rl4co.utils.ops import gather_by_index
from .generator import PDPTWGenerator
from .sf_generator import SFGenerator
from .render import render

class PDPTWEnv(CVRPTWEnv):
    """
    Pickup and Delivery Problem with Time Windows (PDPTW) environment.
    """
    name = "pdptw"

    def __init__(
        self,
        generator: SFGenerator = None,
        generator_params: dict = {},
        depot_penalty: int = 100,
        fleet_count: int = 5,
        **kwargs,
    ):
        self.depot_penalty_factor = int(depot_penalty)
        self.fleet_count = fleet_count
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
        """
        td: Created by calling reset(batchsize) function, 
         .generator returns a tensordict
         This creates a problem instance based on outputs of SFGEnerator;
         By problem instance: It has dynamic fields that are set to zero. 
         Static fields (i.e. locations, distance matrix) are taken from SFGenerator. 
        """

        device = td.device
        
        # Static fields
        h3_indices = td["h3_indices"]
        demands = td["demand"]
        tws = td["time_windows"]
        travel_time_matrix = td["travel_time_matrix"]
        locs = td["locs"]
        flexibility = td["flexibility"] #! Should you add flexibility?

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
        # Gather H3 indices (keep dimensions [B, N])
        from_h3 = gather_by_index(td["h3_indices"], from_node_idx, squeeze=False)
        to_h3 = gather_by_index(td["h3_indices"], to_node_idx, squeeze=False)
        
        # Get rows corresponding to from_nodes: [B, N_from, NumH3]
        # Use squeeze=False to preserve the N_from dimension even if it is 1
        rows = gather_by_index(td["travel_time_matrix"], from_h3, dim=1, squeeze=False) 
        
        n_from = from_h3.size(1)
        n_to = to_h3.size(1)
        
        # Handle broadcasting
        if n_from == 1 and n_to > 1:
            # One source, multiple destinations: Broadcast rows
            rows = rows.expand(-1, n_to, -1)
        elif n_from > 1 and n_to == 1:
            # Multiple sources, one destination: Broadcast target index
            to_h3 = to_h3.expand(-1, n_from)
            
        # Gather columns corresponding to to_nodes
        return gather_by_index(rows, to_h3, dim=2)

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]
        batch_size = action.shape[0]

        curr_node = td["current_node"]
        
        travel_time = self._get_travel_time(td, curr_node, action.unsqueeze(-1))
        
        arrival_time = td["current_time"] + travel_time
        start_window = gather_by_index(td["time_windows"], action)[..., 0]
        service_start_time = torch.max(arrival_time, start_window.unsqueeze(-1))

        is_return_to_depot = (action == 0) & (curr_node.squeeze(-1) != 0)
        service_start_time = torch.where(is_return_to_depot.unsqueeze(-1), torch.zeros_like(service_start_time), service_start_time)
        
        selected_demand = gather_by_index(td["demand"], action).unsqueeze(-1)
        new_load = td["used_capacity"] + selected_demand
        new_load = torch.where(is_return_to_depot.unsqueeze(-1), torch.zeros_like(new_load), new_load)

        new_visited = td["visited"].clone()
        non_depot_mask = (action != 0).unsqueeze(-1)
        new_visited.scatter_(-1, action.unsqueeze(-1), non_depot_mask)

        # --- Graph Update ---
        pending_schedule = td["pending_schedule"].clone()
        pending_count = td["pending_count"].clone()
        is_pickup = (action % 2 != 0) & (action != 0)
        
        # Remove Visited
        mask_in_schedule = (pending_schedule == action.unsqueeze(-1))
        pending_schedule[mask_in_schedule] = 0
        is_dropoff = (action % 2 == 0) & (action != 0)
        pending_count = torch.clamp(pending_count - is_dropoff.long().unsqueeze(-1), min=0)

        # Insert Partner
        num_nodes = td["h3_indices"].shape[1]
        partner_node = torch.clamp(action + 1, max=num_nodes - 1)
        new_entry = torch.zeros((batch_size, 1), dtype=torch.int64, device=td.device)
        new_entry[is_pickup, 0] = partner_node[is_pickup]
        combined = torch.cat([pending_schedule, new_entry], dim=1)
        
        # Sort by TW Late
        node_tws = gather_by_index(td["time_windows"], combined)[..., 1]
        is_zero = (combined == 0)
        node_tws[is_zero] = 1e9 
        _, sort_idx = torch.sort(node_tws, dim=1)
        sorted_sched = torch.gather(combined, 1, sort_idx)
        
        capacity = td["vehicle_capacity"].shape[1] if len(td["vehicle_capacity"].shape)>1 else td["vehicle_capacity"][0].item()
        pending_schedule = sorted_sched[:, :int(capacity)]
        pending_count = torch.clamp(pending_count + is_pickup.long().unsqueeze(-1), max=int(capacity))

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

        # --- 1. Basic Masks ---
        visited = td["visited"]
        current_node = td["current_node"].squeeze(-1)
        current_time = td["current_time"]
        
        node_indices = torch.arange(num_nodes, device=device).expand(batch_size, num_nodes)
        is_depot = (node_indices == 0)
        is_pickup = (node_indices % 2 != 0) & (~is_depot)
        is_dropoff = (node_indices % 2 == 0) & (~is_depot)

        pickup_mask = ~visited & is_pickup
        partner_indices = torch.clamp(node_indices - 1, min=0)
        partner_visited = torch.gather(visited, 1, partner_indices)
        dropoff_mask = partner_visited & (~visited) & is_dropoff
        
        all_visited = visited[..., 1:].all(dim=-1, keepdim=True)
        depot_mask = all_visited & is_depot
        precedence_mask = pickup_mask | dropoff_mask | depot_mask

        remaining_cap = td["vehicle_capacity"] - td["used_capacity"]
        demand_ok = (td["demand"] <= remaining_cap) | is_dropoff | is_depot
        
        travel_time_to_candidate = self._get_travel_time(td, current_node.unsqueeze(-1), node_indices)
        arrival_at_candidate = current_time + travel_time_to_candidate
        
        candidate_late = td["time_windows"][..., 1]
        candidate_early = td["time_windows"][..., 0]
        immediate_tw_mask = (arrival_at_candidate <= candidate_late)

        basic_mask = precedence_mask & demand_ok & immediate_tw_mask

        # --- 2. Advanced Mask: Graph Lookahead ---
        schedule = td["pending_schedule"]
        g0_idx = schedule[:, 0]
        
        # Handle capacity < 2
        if schedule.shape[1] >= 2:
            g1_idx = schedule[:, 1]
        else:
            g1_idx = torch.zeros_like(g0_idx)

        # Safely gather and reshape Late TWs to [B, 1]
        # We gather, then explicitly view/reshape to avoid broadcasting errors if squeeze happens
        g0_late_raw = gather_by_index(td["time_windows"], g0_idx.unsqueeze(-1))[..., 1]
        g1_late_raw = gather_by_index(td["time_windows"], g1_idx.unsqueeze(-1))[..., 1]
        
        g0_late = g0_late_raw.view(batch_size, 1)
        g1_late = g1_late_raw.view(batch_size, 1)

        # Calculate travel time from candidates to g0 and g1
        time_to_g0 = self._get_travel_time(td, node_indices, g0_idx.unsqueeze(-1)) # [B, N]
        time_to_g1 = self._get_travel_time(td, node_indices, g1_idx.unsqueeze(-1)) # [B, N]
        
        departure_from_candidate = torch.max(arrival_at_candidate, candidate_early) # [B, N]
        
        can_reach_g0 = (departure_from_candidate + time_to_g0) <= g0_late # [B, N] vs [B, 1]
        can_reach_g1 = (departure_from_candidate + time_to_g1) <= g1_late
        
        is_g0 = (node_indices == g0_idx.unsqueeze(-1))
        graph_constraint_mask = torch.where(is_g0, can_reach_g1, can_reach_g0)
        
        final_mask = basic_mask & graph_constraint_mask

        # Fallbacks
        schedule_empty = (td["pending_count"] == 0)
        no_feasible = ~final_mask[..., 1:].any(dim=-1, keepdim=True)
        final_mask[..., 0] = final_mask[..., 0] | (no_feasible & schedule_empty).squeeze(-1)
        none_at_all = ~final_mask.any(dim=-1, keepdim=True)
        # Always allow depot as escape hatch when no actions are feasible
        # This prevents "infeasible action selected" errors
        final_mask[..., 0] = final_mask[..., 0] | none_at_all.squeeze(-1)

        return final_mask

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        # Calculate total travel time
        ordered_nodes = torch.cat(
            [
                torch.zeros_like(actions[:, 0:1]), # Depot start
                actions
            ],
            dim=1
        )
        
        # Ordered nodes: [B, N+1]
        from_nodes = ordered_nodes[:, :-1]
        to_nodes = ordered_nodes[:, 1:]
        
        travel_times = self._get_travel_time(td, from_nodes, to_nodes) # [B, N]
        
        return -travel_times.sum(dim=1)

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        pass 

    @staticmethod
    def render(td, actions=None, ax=None):
        return render(td, actions, ax)
