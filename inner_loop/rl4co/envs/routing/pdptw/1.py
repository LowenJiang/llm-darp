from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Unbounded, Bounded, Composite

from inner_loop.rl4co.envs.routing.cvrp.env import CVRPEnv
from inner_loop.rl4co.envs.routing.cvrptw.env import CVRPTWEnv
from inner_loop.rl4co.utils.ops import gather_by_index, get_geo_distance, get_geod_tour_length
from .generator import PDPTWGenerator
from .sf_generator import SFGenerator
from .render import render

class PDPTWEnv(CVRPTWEnv):
    """
    Pickup and Delivery Problem with Time Windows (PDPTW) environment.
    
    Key Features:
    - Precedence Constraints: Pickup (odd) -> Delivery (even).
    - Pending Schedule: Maintains a 'graph' of pending deliveries.
    - Lookahead Masking: Checks if actions allow the remaining schedule to be feasible.
    """

    name = "pdptw"


    def __init__(
        self,
        generator: SFGenerator = None,
        generator_params: dict = {},
        depot_penalty: int = 100, #Penalty for each additioanl vehicle
        fleet_count: int = 5,     #maximum number of vehicles used
        vehicle_speed: float = 0.3,
        late_penalty_factor: float = 1.0,
        **kwargs,
    ):
        # Set attributes BEFORE super().__init__() so they're available in _make_spec
        self.depot_penalty_factor = int(depot_penalty)
        self.fleet_count = fleet_count
        self.vehicle_speed = vehicle_speed
        self.late_penalty_factor = late_penalty_factor
        # Create generator BEFORE super().__init__()
        if generator is None:
            generator = SFGenerator(**generator_params)

        # Pass generator to parent class so it uses the correct one
        super().__init__(generator=generator, **kwargs)

        self.generator = generator
        self._make_spec(self.generator)

    def _make_spec(self, generator):
        """Define observations and action specs."""
        self.observation_spec = Composite(
            locs=Bounded(low=-180, high=180, shape=(generator.num_customers * 2 + 1, 2), dtype=torch.float32),
            time_windows=Bounded(low=0, high=1440, shape=(generator.num_customers * 2 + 1, 2), dtype=torch.float32),
            demand=Bounded(low=-generator.vehicle_capacity, high=generator.vehicle_capacity, shape=(generator.num_customers * 2 + 1,), dtype=torch.float32),
            current_node=Unbounded(shape=(1), dtype=torch.int64),
            current_time=Bounded(low=0, high=1440, shape=(1), dtype=torch.float32),
            used_capacity=Bounded(low=0, high=generator.vehicle_capacity, shape=(1), dtype=torch.float32),
            
            # The 'Graph': Pending deliveries. Size is max capacity (worst case)
            pending_schedule=Bounded(low=0, high=generator.num_customers * 2, shape=(generator.vehicle_capacity,), dtype=torch.int64),
            pending_count=Bounded(low=0, high=generator.vehicle_capacity, shape=(1), dtype=torch.int64),
            
            visited=Bounded(low=0, high=1, shape=(generator.num_customers * 2 + 1,), dtype=torch.bool),
            action_mask=Bounded(low=0, high=1, shape=(generator.num_customers * 2 + 1,), dtype=torch.bool),
        )

        self.action_spec = Bounded(
            shape=(1,), dtype=torch.int64, low=0, high=generator.num_customers * 2 + 1
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        
        # Setup basic CVRP state
        num_locs = td["locs"].shape[-2]
        # Add depot to locs if not handled by generator (RL4CO usually handles this, assuming inputs are proper)
        # If inputs lack depot, we concatenate. Assuming standard RL4CO format where locs excludes depot in generator output sometimes.
        if td["locs"].shape[-2] == self.generator.num_customers * 2:
             locs = torch.cat((td["depot"][..., None, :], td["locs"]), -2)
             demands = torch.cat((torch.zeros_like(td["demand"][..., :1]), td["demand"]), -1)
             tws = torch.cat((torch.zeros_like(td["time_windows"][..., :1, :]), td["time_windows"]), -2)
        else:
             locs = td["locs"]
             demands = td["demand"]
             tws = td["time_windows"]

        # Initialize Pending Schedule (The Graph)
        # Shape: [Batch, Capacity]. Fill with 0 (Depot, acting as padding)
        capacity_limit = self.generator.vehicle_capacity
        pending_schedule = torch.zeros((*batch_size, capacity_limit), dtype=torch.int64, device=device)
        pending_count = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        td_reset = TensorDict({
            "locs": locs,
            "time_windows": tws,
            "demand": demands,
            "current_node": torch.zeros(*batch_size, 1, dtype=torch.int64, device=device),
            "current_time": torch.zeros(*batch_size, 1, dtype=torch.float32, device=device),
            "used_capacity": torch.zeros(*batch_size, 1, dtype=torch.float32, device=device),
            "visited": torch.zeros(*batch_size, locs.shape[-2], dtype=torch.bool, device=device),
            "vehicle_capacity": torch.full((*batch_size, 1), capacity_limit, device=device),
            "vehicle_speed": torch.full((*batch_size, 1), self.vehicle_speed, device=device),
            
            # Graph State
            "pending_schedule": pending_schedule,
            "pending_count": pending_count,
            
            "vehicle_penalty": torch.zeros(*batch_size, 1, dtype=torch.float32, device=device),
        }, batch_size=batch_size)

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"] # [Batch]
        
        # --- 1. Standard Updates (Loc, Time, Capacity, Visited) ---
        curr_node = td["current_node"]
        curr_loc = gather_by_index(td["locs"], curr_node)
        next_loc = gather_by_index(td["locs"], action)
        
        dist = get_geo_distance(curr_loc, next_loc).unsqueeze(-1) # [B, 1]
        travel_time = dist / td["vehicle_speed"]
        
        arrival_time = td["current_time"] + travel_time
        # Wait if early
        start_window = gather_by_index(td["time_windows"], action)[..., 0]
        service_start_time = torch.max(arrival_time, start_window.unsqueeze(-1))
        
        # Update Load
        selected_demand = gather_by_index(td["demand"], action).unsqueeze(-1)
        new_load = td["used_capacity"] + selected_demand

        # Update Visited
        # scatter_ is in-place, be careful with cloning if needed, but RL4CO handles TD copies
        new_visited = td["visited"].clone()
        new_visited.scatter_(-1, action.unsqueeze(-1), 1)

        # --- 2. Update Pending Schedule (The "Graph") ---
        pending_schedule = td["pending_schedule"].clone()
        pending_count = td["pending_count"].clone()

        # Mask for types
        is_pickup = (action % 2 != 0) & (action != 0)
        is_dropoff = (action % 2 == 0) & (action != 0)
        
        # CASE A: Dropoff -> Remove from schedule
        # We find where the action is in the schedule and zero it out, then shift (compact)
        # For vectorization simplicity, we just mask it out now and sort/compact later if needed.
        # However, simply setting to 0 (depot) works if we treat 0 as empty.
        # But we need to maintain order for the heuristic update. 
        
        # Simple removal:
        mask_in_schedule = (pending_schedule == action.unsqueeze(-1))
        pending_schedule[mask_in_schedule] = 0 # Mark as removed
        pending_count -= is_dropoff.long().unsqueeze(-1)

        # CASE B: Pickup -> Add partner (action + 1) to schedule
        partner_node = action + 1
        
        # We need to insert the partner node. 
        # Heuristic: Insert based on Late Time Window (EDD) to keep the schedule "smart".
        # 1. Get TW late for the new partner
        partner_tw_late = gather_by_index(td["time_windows"], partner_node)[..., 1]
        
        # 2. We construct a new list: Current pending (excluding removed) + New Partner (if pickup)
        #    Then we sort this list by TW Late to form the new valid schedule for the next step.
        
        # Create a temporary tensor with the new node added at the end (or 0 if not pickup)
        # Since we can't easily append variably, we replace the first 0 found or use a fixed strategy.
        # Strategy:
        # 1. Collect all nodes currently in schedule (zeros are padding)
        # 2. Add new node if pickup
        # 3. Sort by TW Late (pushing zeros to the end)
        
        # Get all TWs for nodes in schedule. 0 (depot) has huge TW or 0. 
        # We want 0s to go to end. Let's give 0s a generic TW of infinity for sorting.
        
        # [B, C]
        sched_nodes = pending_schedule
        # If pickup, put the partner in one of the empty slots (zeros)
        # Find first zero index? Hard to vectorize efficiently. 
        # Alternative: Add to a specific column if we know count?
        # Let's use a scatter add approach.
        
        # Add partner to the 'end' (or replace a 0). 
        # Since we removed the dropoff (set to 0), we have space. 
        # If pickup, we replaced a 0 with a number.
        
        # Let's brute force the update via sorting:
        # Add candidate to a spare column (we need C+1 temp space or assume capacity constraint holds)
        # Actually, capacity check ensures we don't exceed.
        
        # Insert partner where there is a 0
        # Find empty slots
        is_slot_empty = (pending_schedule == 0)
        # Fill ONLY the first empty slot with partner_node where is_pickup is True
        # This is tricky in pure torch without loop. 
        # Workaround: Create a tensor of just the new node, concat, then top-k/sort.
        
        batch_size, cap = pending_schedule.shape
        new_entry = torch.zeros((batch_size, 1), dtype=torch.int64, device=td.device)
        new_entry[is_pickup] = partner_node[is_pickup]
        
        # Concatenate [Schedule, NewEntry] -> [B, C+1]
        combined = torch.cat([pending_schedule, new_entry], dim=1)
        
        # Get sorting keys: TW Late. 
        # 0s should be at the end.
        node_tws = gather_by_index(td["time_windows"], combined)[..., 1] # [B, C+1]
        
        # Mask 0s to have very high TW so they sort to end
        is_zero = (combined == 0)
        node_tws[is_zero] = 1e9
        
        # Sort indices
        _, sort_idx = torch.sort(node_tws, dim=1)
        
        # Gather sorted nodes
        sorted_sched = torch.gather(combined, 1, sort_idx)
        
        # Keep only first C elements (remove the padding/extra 0 we added if it bubbled down)
        pending_schedule = sorted_sched[:, :cap]
        
        pending_count += is_pickup.long().unsqueeze(-1)

        # --- 3. Finish Step ---
        td_out = td.clone()
        td_out.update({
            "current_node": action.unsqueeze(-1),
            "current_time": service_start_time,
            "used_capacity": new_load,
            "visited": new_visited,
            "pending_schedule": pending_schedule,
            "pending_count": pending_count
        })
        
        # Done?
        # Done if at depot (0) and all requests served.
        # In PDPTW, usually done when returned to depot AND visited all.
        all_visited = new_visited[..., 1:].all() # excluding depot
        is_depot = (action == 0)
        done = is_depot & all_visited # Simplified done condition
        
        td_out.set("done", done)
        
        # Compute Mask for next step
        td_out.set("action_mask", self.get_action_mask(td_out))
        
        return td_out

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Implement the pseudocode _get_mask logic.
        """
        batch_size = td["locs"].shape[0]
        num_nodes = td["locs"].shape[1]
        
        # --- 1. Basic Masking ---
        
        # Unpack
        visited = td["visited"]
        current_node = td["current_node"].squeeze(-1)
        current_time = td["current_time"]
        used_cap = td["used_capacity"]
        capacity = td["vehicle_capacity"]
        remaining_cap = capacity - used_cap
        demands = td["demand"]
        
        # A. Precedence
        # Pickup (Odd): Not visited
        # Dropoff (Even): Partner (node-1) visited, Self not visited
        
        # Generate indices [1, ... N]
        # We process all nodes at once via broadcasting
        # Node indices: [0, 1, 2, ... ]
        node_indices = torch.arange(num_nodes, device=td.device).expand(batch_size, num_nodes)
        
        is_depot = (node_indices == 0)
        is_pickup = (node_indices % 2 != 0) & (~is_depot)
        is_dropoff = (node_indices % 2 == 0) & (~is_depot)
        
        # Piekcup mask: not visited
        pickup_mask = ~visited & is_pickup
        
        # Dropoff mask: partner visited AND not visited
        partner_indices = node_indices - 1
        # Gather visited status of partners
        partner_visited = torch.gather(visited, 1, partner_indices)
        dropoff_mask = partner_visited & (~visited) & is_dropoff
        
        # Depot mask: Valid if everyone else visited (or end of episode logic)
        # For now, let's say depot is valid if schedule is empty? Or standard RL4CO logic:
        # usually valid if all customers visited.
        all_customers_visited = visited[..., 1:].all(dim=-1, keepdim=True)
        depot_mask = all_customers_visited & is_depot
        
        precedence_mask = pickup_mask | dropoff_mask | depot_mask
        
        # B. Capacity
        # Only Pickups consume capacity. Dropoffs free it (always feasible load-wise).
        # Check: demand[node] <= remaining_cap
        node_demands = demands # [B, N]
        cap_feasible = (node_demands <= remaining_cap)
        # For dropoffs, demand is negative, so used_cap + demand < capacity is trivial.
        # For pickups, demand is positive.
        capacity_mask = cap_feasible | (~is_pickup) # True if dropoff or depot or feasible pickup
        
        # C. Immediate Time Window
        # Can we get to node before Late TW?
        # Travel from current to all nodes
        curr_loc = gather_by_index(td["locs"], current_node.unsqueeze(-1)) # [B, 1, 2]
        dest_locs = td["locs"] # [B, N, 2]
        
        dists = get_geo_distance(curr_loc, dest_locs) # [B, N]
        travel_times = dists / td["vehicle_speed"]
        arrival_times = current_time + travel_times
        
        # Check Late Window
        late_windows = td["time_windows"][..., 1]
        time_mask = (arrival_times <= late_windows)
        
        # Combine Basic Masks
        basic_mask = precedence_mask & capacity_mask & time_mask
        
        # If basic mask is all false, return (early exit optimization not easily done in batch)
        # We continue to advanced masking, applying it only where basic_mask is True.
        
        # --- 2. Advanced Masking (Graph Feasibility) ---
        
        # This part is computationally heavy. We only check candidates that passed basic mask.
        # To vectorize, we iterate over the *types* of checks.
        
        graph = td["pending_schedule"] # [B, C]
        
        # Initialize advanced_mask as True (default) or False? 
        # Pseudocode: "If graph empty... True". "Else... check".
        # Let's start with False, and set to True if checks pass. 
        # Note: If graph is empty, checks should trivially pass if we implement correctly.
        advanced_mask = torch.zeros_like(basic_mask)
        
        # Optimization: If graph is empty, all basic valid moves are valid.
        graph_is_empty = (td["pending_count"] == 0)
        advanced_mask[graph_is_empty.squeeze(-1)] = True 
        
        # We only need to check rows where graph is NOT empty and basic_mask is True
        # But for batch consistency, we run checks for all (masking out invalid results later).
        
        # === Check A: Dropoffs ===
        # Candidate `new_node` is a dropoff.
        # Seq: [new_node] + [graph - new_node]
        # Logic: Current -> new_node -> (Graph without new_node)
        
        # 1. Construct the sequence for Dropoff Candidates
        # Since different candidates result in different "Remaining Graphs", 
        # we might need to loop over the *index in the graph* or just the candidate node index.
        # Since capacity is small, let's loop over the `graph` columns. 
        # A candidate dropoff MUST be in the graph.
        
        # Iterating over potential dropoff actions is cleaner:
        # Filter columns where `basic_mask & is_dropoff` is True.
        
        # To vectorize efficiently:
        # We check the feasibility of the sequence: Current -> Graph_Node_i -> Graph_Nodes_excluding_i
        # for every i in [0, Capacity].
        
        C = graph.shape[1]
        
        for i in range(C):
            target_node = graph[:, i] # [B] - The candidate dropoff at index i
            
            # Only valid if target_node is not 0 (padding)
            valid_target = (target_node != 0)
            
            # Create sequence: [target_node, graph without i]
            # Construct tensor [B, C]
            # We can just take `graph` and swap index `i` to the front? 
            # Actually, `check_path` takes a sequence. 
            # Sequence: target_node (1) + remaining (C-1)
            
            # Mask out i from graph
            # We can roll or concat.
            remaining = torch.cat([graph[:, :i], graph[:, i+1:]], dim=1) # [B, C-1]
            seq = torch.cat([target_node.unsqueeze(1), remaining], dim=1) # [B, C]
            
            # Check Feasibility
            is_feasible = self._check_path_feasible(td, current_node, current_time, seq)
            
            # Update mask for this specific node
            # We need to map `target_node` (the action index) back to the mask index
            # scatter check
            # advanced_mask[b, target_node[b]] |= is_feasible[b]
            
            # We scatter only where valid_target is True
            action_idx = target_node.unsqueeze(1) # [B, 1]
            # Only update if action_idx is valid action (not 0/padding)
            
            update_val = is_feasible.unsqueeze(1) & valid_target.unsqueeze(1)
            advanced_mask.scatter_(1, action_idx, update_val, reduce='add') # logical OR via add on 0/1? 
            # scatter does not support 'or'. Use max or manual loop.
            # Since we init 0, max works.
            # However, multiple slots might point to same node (unlikely in valid graph)? No, unique nodes.
            
            # Workaround for scatter OR:
            # We can just use boolean indexing if we weren't batching.
            # Let's use a temporary mask.
            batch_idx = torch.arange(batch_size, device=td.device)
            
            # Apply only to valid batches
            valid_b = valid_target
            if valid_b.any():
                b_indices = batch_idx[valid_b]
                a_indices = target_node[valid_b]
                vals = is_feasible[valid_b]
                advanced_mask[b_indices, a_indices] = (advanced_mask[b_indices, a_indices] | vals)

        # === Check B: Pickups ===
        # Candidate `new_node` is a pickup.
        # Partner = new_node + 1.
        # We need to find *at least one* insertion spot k in [0..len(graph)] 
        # such that [new_node] + graph[:k] + [partner] + graph[k:] is valid.
        
        # This implies: For a specific Pickup Candidate, is there ANY valid future schedule?
        
        # This is expensive: [N_pickups] x [Capacity+1 insertions] x [Path Checks].
        # We can invert loops:
        # Loop over insertion index k [0..C].
        #   Construct Template Sequence: Graph[:k] + [PartnerPlaceholder] + Graph[k:]
        #   This template depends on the *Partner*.
        #   Check feasibility for *all pickup candidates* in parallel against this template.
        
        # 1. Identify Pickup Candidates (from basic mask)
        # We perform checks for ALL pickups, then mask result.
        
        # Sequence for insertion at k:
        # S = [Candidate] + Graph[:k] + [Candidate+1] + Graph[k:]
        # Length = 1 + k + 1 + (C-k) = C + 2.
        
        pickup_cands_indices = torch.arange(num_nodes, device=td.device).expand(batch_size, num_nodes)
        is_cand_pickup = (pickup_cands_indices % 2 != 0) & (pickup_cands_indices != 0)
        
        # We need to run the check for every pickup candidate.
        # To vectorize this, we might need to expand dimensions: [B, N, SeqLen]
        # N is roughly 50-100. SeqLen is Capacity (e.g. 5-10). 
        # B*N*C is manageable (e.g. 64*50*10 = 32k operations).
        
        # Let's iterate k (insertion point).
        # For each k, we check feasibility for ALL pickup nodes.
        # If feasible for index k, we mark that pickup as valid.
        
        # Current Graph with 0s at end.
        # Valid length of graph is pending_count. 
        # Insertion index k can range from 0 to pending_count.
        # To vectorize, we just iterate 0 to C. 
        # If k > pending_count, it's effectively appending (since graph[k] is 0).
        # We treat 0s as dummy nodes that have 0 distance and 0 service time? 
        # No, 0 is depot. Visiting depot mid-route is usually bad or ends route.
        # The `_check_path_feasible` function should handle 0s gracefully (skip them or treat as instant).
        # Let's ensure 0s (padding) don't break feasibility. 
        # In `_check_path_feasible`, if node is 0 (and not end of route), we might want to ignore it.
        # Or, simpler: The graph provided to check should simply not have the padding 0s in the middle.
        
        # Let's construct sequences carefully.
        
        # Loop over insertion position `k` from 0 to `capacity`.
        for k in range(C + 1):
            # We want to check: Can we insert Partner at position k of the VALID graph?
            # But "Valid Graph" varies per batch.
            # Simplification: 
            # We construct a sequence `Base_Seq` = Graph with a "Gap" at k.
            # Actually, inserting at k means: Nodes 0..k-1, then Partner, then Nodes k..end.
            
            # Prepare the static parts of the graph for this k
            # pre: graph[:, :k]
            # post: graph[:, k:]
            
            # We need to check `current -> pickup -> pre -> partner -> post`
            
            # We process all Pickups in parallel?
            # Reshape inputs to [B, N_nodes] so we can verify all candidates.
            
            # Current Time/Loc: [B, 1] -> [B, N]
            # Graph parts: [B, k] -> [B, 1, k] -> repeat to [B, N, k]
            
            # This might consume too much memory if N is large. 
            # Loop over N (Pickups) might be safer if OOM, but Loop over k is better.
            # Let's stick to [B, N] computation.
            
            # Construct Test Sequence [B, N, C+2]
            # Dimensions: Batch, Candidate_Pickup, Sequence_Step
            
            # 1. Candidate Node [B, N]
            cands = node_indices # [B, N]
            
            # 2. Pre-Graph [B, k] -> [B, N, k]
            pre_g = graph[:, :k].unsqueeze(1).expand(-1, num_nodes, -1)
            
            # 3. Partner [B, N]
            partners = cands + 1
            
            # 4. Post-Graph [B, C-k] -> [B, N, C-k]
            post_g = graph[:, k:].unsqueeze(1).expand(-1, num_nodes, -1)
            
            # Concat: [Cand, Pre, Partner, Post]
            # Note: If k > pending_count, this logic inserts 0s in between. 
            # We should effectively shift 0s to the end. 
            # However, if we just check the sequence as is, 0 (Depot) usually allows "wait until end", 
            # but distance to 0 and back is costly.
            # To do this correctly vectorized: we shouldn't verify invalid insertions (e.g. inserting after padding).
            # But `_check_path_feasible` will just return False if 0s mess up time, or True if they are innocuous.
            # Ideally we only check if k <= pending_count[b].
            
            seq = torch.cat([
                cands.unsqueeze(-1),
                pre_g,
                partners.unsqueeze(-1),
                post_g
            ], dim=-1) # [B, N, C+2]
            
            # Mask: We only care about rows where `cands` is a Pickup
            # and `k <= pending_count`.
            valid_insertion_spot = (k <= td["pending_count"]) # [B, 1]
            check_mask = is_cand_pickup & valid_insertion_spot
            
            if not check_mask.any():
                continue
                
            # Run Check
            # _check_path_feasible needs to handle [B, N, Seq] input
            is_valid = self._check_path_feasible(td, current_node, current_time, seq, num_candidates=num_nodes)
            
            # Accumulate Result
            # If valid for this k, then advanced_mask |= True
            advanced_mask[check_mask & is_valid] = True

        # Combine
        # For pickups: basic & advanced
        # For dropoffs: basic & advanced
        # For depot: basic (advanced check assumed True or handled separately)
        
        # Re-assert graph_is_empty logic just in case
        final_mask = basic_mask & advanced_mask
        
        # Fix Depot: Depot is valid if graph is empty (or all done).
        # Basic mask handled `all_customers_visited`.
        # If graph is not empty, depot is typically invalid in PDPTW unless we allow dumping (not standard).
        # So `depot_mask` inside `basic_mask` already checks `all_customers_visited`.
        # If `all_customers_visited`, graph must be empty.
        # So just `basic_mask` for depot is fine.
        
        # Apply "Or Graph Empty" fallback? 
        # If graph is empty, advanced_mask is True. So basic & advanced = basic. Correct.
        
        return final_mask

    def _check_path_feasible(self, td, start_node, start_time, sequence, num_candidates=None):
        """
        Vectorized check of path feasibility.
        sequence: [B, SeqLen] OR [B, N_candidates, SeqLen]
        start_node: [B]
        start_time: [B, 1]
        """
        # Normalize inputs to [Batch * Candidates, SeqLen] for flattened processing
        # or keep [B, N, S] and use broadcasting.
        
        is_3d = (sequence.dim() == 3)
        
        if is_3d:
            batch_size, n_cands, seq_len = sequence.shape
            # Expand static context
            # locs: [B, N_locs, 2] -> [B, 1, N_locs, 2]
            locs = td["locs"].unsqueeze(1)
            # time_windows: [B, 1, N_locs, 2]
            tws = td["time_windows"].unsqueeze(1)
            
            # Current state
            curr_loc = gather_by_index(td["locs"], start_node).unsqueeze(1) # [B, 1, 2]
            curr_time = start_time.unsqueeze(1) # [B, 1, 1]
            
        else:
            batch_size, seq_len = sequence.shape
            locs = td["locs"]
            tws = td["time_windows"]
            curr_loc = gather_by_index(td["locs"], start_node)
            curr_time = start_time
            n_cands = 1

        # Iterate through sequence
        valid = torch.ones(batch_size, n_cands, dtype=torch.bool, device=td.device)
        if not is_3d: valid = valid.squeeze(-1)

        # We step through the sequence columns
        t = curr_time
        loc = curr_loc
        
        for i in range(seq_len):
            node_idx = sequence[..., i] # [B, N] or [B]
            
            # Handle padding 0s:
            # If node is 0 (Depot), it usually has wide windows.
            # But if it's padding *after* real nodes, we shouldn't incur distance cost from last node to depot 
            # unless it's the final return. 
            # Simplification: Just calculate. If 0 is padding, it's effectively "stay at depot" or "go to depot".
            # If we are in the middle of a route, going to depot is usually invalid if not empty load.
            # However, this function just checks TIME.
            
            target_loc = gather_by_index(locs, node_idx) # [B, N, 2]
            target_tw = gather_by_index(tws, node_idx)   # [B, N, 2]
            early = target_tw[..., 0]
            late = target_tw[..., 1]
            
            # Dist
            dist = get_geo_distance(loc, target_loc)
            tt = dist / td["vehicle_speed"]
            if is_3d: tt = tt.unsqueeze(-1) # match dims
            
            arrival = t + tt
            
            # Check Late
            # If node is 0 (padding) and we are just checking feasibility, 
            # we assume 0s at the end of sequence don't fail (TW is 0-1440).
            # However, going back and forth to depot 0 0 0 0 drains time.
            # Mask out checks for 0s if they are padding?
            # If we properly constructed sequence, 0s are at end. 
            # Let's assume standard constraints apply.
            
            is_late = (arrival > late.unsqueeze(-1) if is_3d else arrival > late)
            
            # Update Valid
            valid = valid & (~is_late)
            
            # Update Time (Wait if early)
            t = torch.max(arrival, early.unsqueeze(-1) if is_3d else early)
            
            # Update Loc
            loc = target_loc
        
        return valid.squeeze(-1) if is_3d else valid