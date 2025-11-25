from typing import Optional
import warnings

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from inner_loop.rl4co.envs.common.base import RL4COEnvBase
from inner_loop.rl4co.utils.ops import gather_by_index

from .generator import DARPGenerator
from .render import render
from .utils import calculate_successful_requests


class DARPEnv(RL4COEnvBase):
    """Dial-a-Ride Problem (DARP) environment.

    The environment consists of:
        - 1 depot
        - num_loc / 2 pickup locations
        - num_loc / 2 dropoff locations
        - num_agents vehicles (each vehicle makes ONE tour)

    The goal is to serve all pickup-dropoff requests while minimizing total distance traveled.

    Observations:
        - locs: locations of all nodes [num_loc + 1, 2] (depot + customers)
        - current_node: current node of the active vehicle [1]
        - current_agent: index of currently active vehicle [1]
        - current_time: current time for the active vehicle [1]
        - current_load: current load for the active vehicle [1]
        - visited: whether each node has been visited by ANY vehicle [num_loc + 1]
        - picked_up_by_current: which pickups have been picked up by current vehicle [num_loc + 1]
        - vehicle_finished: which vehicles have completed their tours [num_agents]
        - i: current step [1]
        - action_mask: mask of available actions [num_loc + 1]

    Constraints:
        - Pickup must precede dropoff for each request
        - Pickup and dropoff must be served by the same vehicle
        - Time window constraints must be satisfied
        - Vehicle capacity cannot be exceeded
        - Each vehicle can return to depot only once (ends their tour)

    Finish Condition:
        - All nodes visited OR all vehicles have finished their tours

    Reward:
        - Negative of (total distance + penalty * num_unvisited_nodes)

    Args:
        generator: DARPGenerator instance as the data generator
        generator_params: parameters for the generator
        penalty_unvisited: penalty coefficient for unvisited nodes (default: 100)
    """
    
    name = "darp"
    
    def __init__(
        self,
        generator: DARPGenerator = None,
        generator_params: dict = {},
        penalty_unvisited: float = 100.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = DARPGenerator(**generator_params)
        self.generator = generator
        self.penalty_unvisited = penalty_unvisited
        self._make_spec(self.generator)
    
    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]

        # Ensure current_node has correct shape for scatter operations
        if current_node.dim() == 1:
            current_node = current_node.unsqueeze(-1)

        # Track depot self-loops (staying at depot)
        prev_at_depot = td["current_node"] == 0
        now_selecting_depot = current_node == 0
        depot_self_loop = prev_at_depot & now_selecting_depot

        num_agents = td["capacity"].shape[-1]
        num_loc = td["locs"].shape[-2] - 1  # excluding depot
        batch_size = td.batch_size

        # Update visited status (global - any vehicle)
        visited = td["visited"].scatter(-1, current_node, 1)

        # Calculate travel time to the new node
        prev_loc = gather_by_index(td["locs"], td["current_node"])
        curr_loc = gather_by_index(td["locs"], current_node)
        dist = torch.norm(curr_loc - prev_loc, p=2, dim=-1, keepdim=True)
        travel_time = torch.round(dist / self.generator.vehicle_speed).long()

        # Update current time
        current_time = td["current_time"] + travel_time

        # Update load and picked_up_by_current status
        current_load = td["current_load"].clone()
        picked_up_by_current = td["picked_up_by_current"].clone()

        # Interleaved nodes: depot=0, pickups at odd indices, dropoffs at even indices
        is_pickup = (current_node % 2 == 1)
        is_dropoff = (current_node % 2 == 0) & (current_node > 0)

        # If pickup: increase load and mark as picked up by current vehicle
        if is_pickup.any():
            demand = gather_by_index(td["demand"], current_node).unsqueeze(-1)
            current_load = torch.where(is_pickup, current_load + demand.long(), current_load)
            pickup_flags = is_pickup.to(picked_up_by_current.dtype)
            picked_up_by_current = picked_up_by_current.scatter(-1, current_node, pickup_flags)

        # If dropoff: decrease load and clear the corresponding pickup from picked_up_by_current
        if is_dropoff.any():
            demand = gather_by_index(td["demand"], current_node).unsqueeze(-1)
            current_load = torch.where(is_dropoff, current_load + demand.long(), current_load)  # demand is negative for dropoff

            # Clear the corresponding pickup (current_node - 1) from picked_up_by_current
            # This indicates the pickup-dropoff pair is now complete for this vehicle
            # Only for actual dropoff nodes (even indices > 0)
            corresponding_pickup = (current_node - 1).clamp(min=0)  # Prevent negative index
            # Only clear for actual dropoffs (not for depot or pickups)
            should_clear = is_dropoff & (current_node > 0)
            if should_clear.any():
                picked_up_by_current = picked_up_by_current.scatter(-1, corresponding_pickup, 0)

        # Track total distance for reward calculation
        total_distance = td["total_distance"] + dist.squeeze(-1)

        # Check if vehicle is returning to depot (ending tour)
        # Returning = going to depot from non-depot location
        prev_at_depot = td["current_node"] == 0
        now_at_depot = current_node == 0
        returning_to_depot = now_at_depot & ~prev_at_depot

        # Initialize vehicle state tracking
        current_agent = td["current_agent"].clone()
        vehicle_finished = td["vehicle_finished"].clone()

        # Check if there are still unvisited nodes
        unvisited_customers = ~visited[..., 1:]  # Exclude depot
        has_unvisited = unvisited_customers.any(dim=-1, keepdim=True)

        # If vehicle returned to depot, mark it as finished and switch to next available vehicle
        should_switch = returning_to_depot.squeeze(-1) & has_unvisited.squeeze(-1)

        # Mark current vehicle as finished when it returns to depot
        if returning_to_depot.any():
            # Use advanced indexing to mark the current agent as finished
            batch_indices = torch.arange(vehicle_finished.shape[0], device=vehicle_finished.device)
            agent_indices = current_agent.squeeze(-1).long()
            vehicle_finished[batch_indices, agent_indices] = True

        # Find next available (unfinished) vehicle for each batch element
        next_agent = self._get_next_available_vehicle(vehicle_finished, num_agents, current_agent)

        # Switch to next vehicle if needed and available
        has_next_vehicle = next_agent >= 0
        should_switch = should_switch & has_next_vehicle

        # Reset state for new vehicle
        current_agent = torch.where(
            should_switch.unsqueeze(-1),
            next_agent.unsqueeze(-1),
            current_agent
        )
        current_time = torch.where(
            should_switch.unsqueeze(-1),
            torch.zeros_like(current_time),
            current_time
        )
        current_load = torch.where(
            should_switch.unsqueeze(-1),
            torch.zeros_like(current_load),
            current_load
        )
        # Clear pickup state for new vehicle
        picked_up_by_current = torch.where(
            should_switch.unsqueeze(-1),
            torch.zeros_like(picked_up_by_current),
            picked_up_by_current
        )
        # New vehicle starts at depot
        current_node = torch.where(
            should_switch.unsqueeze(-1),
            torch.zeros_like(current_node),
            current_node
        )

        # Generate action mask
        action_mask = self._get_action_mask(
            td.update({
                "current_node": current_node,
                "current_agent": current_agent,
                "current_time": current_time,
                "current_load": current_load,
                "visited": visited,
                "picked_up_by_current": picked_up_by_current,
                "vehicle_finished": vehicle_finished,
            })
        )

        # Check if done: all customers visited OR all vehicles finished
        all_visited = visited[..., 1:].all(dim=-1)
        all_vehicles_finished = vehicle_finished.all(dim=-1)
        done = all_visited | all_vehicles_finished

        # Also mark done if no actions are available (deadlock/infeasible state)
        no_actions_available = ~action_mask.any(dim=-1)
        done = done | no_actions_available

        # Terminate if depot self-loop occurred (infeasible/stuck state)
        # This happens when vehicle is at depot and can only select depot again
        done = done | depot_self_loop.squeeze(-1)

        # Reward is 0 during episode, computed at the end via get_reward
        reward = torch.zeros_like(done, dtype=torch.float32)

        # Update TensorDict
        td.update({
            "current_node": current_node,
            "current_agent": current_agent,
            "current_time": current_time,
            "current_load": current_load,
            "visited": visited,
            "picked_up_by_current": picked_up_by_current,
            "vehicle_finished": vehicle_finished,
            "total_distance": total_distance,
            "i": td["i"] + 1,
            "action_mask": action_mask,
            "done": done,
            "reward": reward,
        })

        return td

    def _get_next_available_vehicle(
        self,
        vehicle_finished: torch.Tensor,
        num_agents: int,
        current_agent: torch.Tensor
    ) -> torch.Tensor:
        """Find next available (unfinished) vehicle for each batch element.

        Args:
            vehicle_finished: [B, V] boolean tensor indicating finished vehicles
            num_agents: number of vehicles
            current_agent: [B, 1] current agent index

        Returns:
            [B] tensor with next available vehicle index, or -1 if none available
        """
        batch_size = vehicle_finished.shape[0]
        device = vehicle_finished.device

        # Initialize with -1 (no vehicle available)
        next_vehicle = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        # For each batch element, find first unfinished vehicle
        for b in range(batch_size):
            for v in range(num_agents):
                if not vehicle_finished[b, v]:
                    next_vehicle[b] = v
                    break

        return next_vehicle
    
    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td["depot"].device

        num_agents = td["capacity"].shape[-1]
        num_loc = td["locs"].shape[-2]

        # Add depot to locations (depot is at index 0)
        locs = torch.cat([td["depot"].unsqueeze(-2), td["locs"]], dim=-2)

        # Initialize state
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        current_agent = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        current_time = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        current_load = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Visited: track which customer nodes have been visited by ANY vehicle
        # Note: depot (index 0) is NOT marked as visited since vehicles can return to it
        visited = torch.zeros((*batch_size, num_loc + 1), dtype=torch.bool, device=device)

        # Picked up by current vehicle: track which pickups have been picked up by current vehicle
        picked_up_by_current = torch.zeros((*batch_size, num_loc + 1), dtype=torch.bool, device=device)

        # Vehicle finished: track which vehicles have completed their tours
        vehicle_finished = torch.zeros((*batch_size, num_agents), dtype=torch.bool, device=device)

        # Total distance traveled
        total_distance = torch.zeros(*batch_size, dtype=torch.float32, device=device)

        # Step counter
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Create time windows tensor including depot (depot has no time window, set to max)
        time_windows = torch.cat([
            torch.full((*batch_size, 1), 48, dtype=torch.long, device=device),
            td["time_windows"]
        ], dim=-1)

        # Create demand tensor including depot (depot has 0 demand)
        demand = torch.cat([
            torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
            td["demand"]
        ], dim=-1)

        # Generate initial action mask
        td_temp = TensorDict({
            "locs": locs,
            "current_node": current_node,
            "current_agent": current_agent,
            "current_time": current_time,
            "current_load": current_load,
            "visited": visited,
            "picked_up_by_current": picked_up_by_current,
            "vehicle_finished": vehicle_finished,
            "time_windows": time_windows,
            "demand": demand,
            "capacity": td["capacity"],
            "i": i,
        }, batch_size=batch_size)

        action_mask = self._get_action_mask(td_temp)

        # Initialize done as False
        done = torch.zeros(*batch_size, dtype=torch.bool, device=device)

        return TensorDict({
            "locs": locs,
            "current_node": current_node,
            "current_agent": current_agent,
            "current_time": current_time,
            "current_load": current_load,
            "visited": visited,
            "picked_up_by_current": picked_up_by_current,
            "vehicle_finished": vehicle_finished,
            "total_distance": total_distance,
            "time_windows": time_windows,
            "demand": demand,
            "capacity": td["capacity"],
            "i": i,
            "action_mask": action_mask,
            "done": done,
        }, batch_size=batch_size)
        
    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Generate action mask based on constraints"""
        # === dimensions ===
        batch_dims = td["locs"].shape[:-2]            # support extra batch dims, e.g., [B] or [B,S]
        num_loc = td["locs"].shape[-2] - 1           # excluding depot
        device = td["locs"].device

        action_mask = torch.zeros((*batch_dims, num_loc + 1), dtype=torch.bool, device=device)

        visited   = td["visited"]                        # [..., N] - global visited
        picked_up_by_current = td["picked_up_by_current"]  # [..., N] - current vehicle's pickups

        # Normalize shapes: ensure [B] for scalars and [B,1] where needed
        current_node  = td["current_node"].view(*batch_dims, -1)[..., :1].long()   # [...,1]
        current_agent = td["current_agent"].view(*batch_dims, -1)[..., :1].long()  # [...,1]
        current_time  = td["current_time"].view(*batch_dims, -1)[..., 0].long()    # [...]
        current_load  = td["current_load"].view(*batch_dims, -1)[..., 0].long()    # [...]
        i_val         = td["i"].view(*batch_dims, -1)[..., 0].long()               # [...]

        # Capacity for current agent: [B]
        capacity = td["capacity"].gather(-1, current_agent.squeeze(-1).unsqueeze(-1)).squeeze(-1)  # [...]

        # Compute distances and arrival times for all nodes at once
        # curr_loc: [B,1,2]; locs: [B,N,2] where N=num_loc+1
        curr_loc = td["locs"].gather(
            -2, current_node.unsqueeze(-1).expand(*current_node.shape, td["locs"].size(-1))
        )  # [...,1,2]
        all_locs = td["locs"]                                # [...,N,2]
        dist = (all_locs - curr_loc).norm(p=2, dim=-1)        # [...,N]
        travel_time = torch.round(dist / self.generator.vehicle_speed).long()  # [...,N]
        arrival_time = current_time.unsqueeze(-1) + travel_time                 # [...,N]

        # Node indices helpers
        N = num_loc + 1
        idx = torch.arange(N, device=device)
        node_broadcast_shape = (1,) * len(batch_dims) + (N,)
        is_pickup_node = (idx % 2 == 1).view(node_broadcast_shape)
        is_dropoff_node = ((idx % 2 == 0) & (idx != 0)).view(node_broadcast_shape)

        # Common feasibility terms
        time_feasible = arrival_time <= td["time_windows"]                  # [...,N]
        not_visited = ~visited                                               # [...,N]

        # Capacity feasibility for pickups
        demand = td["demand"]                                               # [...,N]
        capacity_feasible = (current_load.unsqueeze(-1) + demand) <= capacity.unsqueeze(-1)  # [...,N]

        # Pickup feasibility (mask out non-pickup nodes by AND with is_pickup_node)
        pickup_mask = not_visited & time_feasible & capacity_feasible        # [...,N]
        pickup_mask = pickup_mask & is_pickup_node                           # [...,N]

        # Dropoff feasibility requires corresponding pickup done by CURRENT vehicle
        # For dropoff at even index i, corresponding pickup is at index i-1
        # Use roll to shift: after roll, position i contains value from position i-1
        pickup_done_by_current = torch.roll(picked_up_by_current, shifts=1, dims=-1)  # [...,N]
        pickup_done_by_current[..., 0] = False  # depot has no corresponding pickup

        dropoff_mask = not_visited & time_feasible & pickup_done_by_current  # [...,N]
        dropoff_mask = dropoff_mask & is_dropoff_node                        # [...,N]

        # Combine pickup and dropoff masks
        action_mask = action_mask | pickup_mask | dropoff_mask               # [...,N]

        # Depot logic: Strategic choice to end vehicle's tour
        load_is_zero = (current_load == 0)
        at_depot = (current_node.squeeze(-1) == 0)
        not_at_depot = ~at_depot

        # Check if there are any pending dropoffs (picked up but not yet delivered)
        # A node is "pending dropoff" if: it's picked up by current vehicle but its dropoff hasn't been visited
        has_pending_dropoffs = picked_up_by_current.any(dim=-1)  # Any pickup done means dropoff might be pending

        # Allow depot return (ends tour) if:
        # 1. Not first step (i > 0)
        # 2. Load is zero (completed all picked up deliveries)
        # 3. NOT currently at depot (prevent self-loops, depot return ends tour)
        # 4. NO pending dropoffs (all picked-up nodes have been delivered)
        depot_feasible = (i_val > 0) & load_is_zero & not_at_depot & ~has_pending_dropoffs
        action_mask[..., 0] = depot_feasible

        # Special case: If no other actions are available, allow depot as emergency exit.
        # This handles infeasible instances and ensures there's always at least one action.
        no_customer_actions = ~action_mask[..., 1:].any(dim=-1)

        # Allow depot if no customer actions - this ensures there's always at least one action available
        # Even at the initial state (i=0), if no customer actions are feasible, depot should be available
        depot_emergency = no_customer_actions
        action_mask[..., 0] = action_mask[..., 0] | depot_emergency

        return action_mask

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Public method to get action mask"""
        return self._get_action_mask(td)
    
    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Calculate reward as negative of (total distance + penalty for unvisited/orphaned nodes)

        Penalties:
        1. Unvisited nodes: penalty_unvisited per node
        2. Orphaned dropoffs: 2x penalty_unvisited per node
           (pickup done but dropoff not done - more severe infeasibility)

        Also calculates and stores successful_requests metric in TensorDict.
        """
        visited = td["visited"]

        # Count unvisited customer nodes (excluding depot)
        num_unvisited = (~visited[..., 1:]).sum(dim=-1).float()

        # Detect orphaned dropoffs: pickup visited but corresponding dropoff not visited
        # In DARP, nodes are interleaved: odd indices (1,3,5,...) are pickups, even indices (2,4,6,...) are dropoffs
        # For each dropoff at index 2i, corresponding pickup is at index 2i-1

        num_loc = visited.shape[-1] - 1  # excluding depot
        num_orphaned = torch.zeros_like(num_unvisited)

        # Check each pickup-dropoff pair
        for i in range(1, num_loc, 2):  # i = 1, 3, 5, ... (pickup indices)
            if i + 1 <= num_loc:  # Make sure dropoff index exists
                pickup_visited = visited[..., i]      # Pickup at odd index i
                dropoff_visited = visited[..., i + 1]  # Dropoff at even index i+1

                # Orphaned: pickup visited but dropoff NOT visited
                orphaned = pickup_visited & ~dropoff_visited
                num_orphaned += orphaned.float()

        # Total unvisited excluding orphaned dropoffs (to avoid double counting)
        num_unvisited_only = num_unvisited - num_orphaned

        # Calculate total cost with different penalties:
        # - Regular unvisited: 1x penalty
        # - Orphaned dropoffs: 2x penalty (more severe)
        cost = (td["total_distance"] +
                self.penalty_unvisited * num_unvisited_only +
                2.0 * self.penalty_unvisited * num_orphaned)

        # Calculate successful requests (completed pickup-dropoff pairs)
        successful_requests = calculate_successful_requests(actions)
        td.set("successful_requests", successful_requests.float())

        return -cost

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid:
        - Visited customer nodes appear at most once
        - Pickups precede their corresponding dropoffs
        - Time window constraints are satisfied for visited nodes
        - Vehicle capacity is not exceeded
        - Same vehicle serves pickup-dropoff pair

        Note: This allows partial solutions (not all customers visited) which is
        common when the policy hits max_steps or during early training.
        """
        batch_size = td["locs"].shape[0]
        num_loc = td["locs"].shape[1] - 1  # excluding depot

        # Use default vehicle speed if not available in td
        vehicle_speed = 25.0  # Default value matching generator default

        # Check that visited customers appear at most once (no duplicates)
        for b in range(batch_size):
            customer_actions = actions[b][actions[b] > 0]  # Filter out depot
            unique_customers = torch.unique(customer_actions)
            if len(unique_customers) != len(customer_actions):
                warnings.warn(
                    f"Customer nodes visited more than once in batch {b}",
                    RuntimeWarning,
                )

        # Check constraints per batch
        for b in range(batch_size):
            action_seq = actions[b]

            # Track state for each vehicle tour
            current_load = 0
            current_time = 0
            current_node = 0
            picked_up_in_tour = set()

            for step, next_node in enumerate(action_seq):
                next_node = next_node.item()

                # Calculate travel
                curr_loc = td["locs"][b, current_node]
                next_loc = td["locs"][b, next_node]
                dist = torch.norm(next_loc - curr_loc, p=2).item()
                travel_time = round(dist / vehicle_speed)
                current_time += travel_time

                if next_node == 0:  # Return to depot
                    # With the fix in action masking, this should only happen when load is 0
                    # and all picked-up nodes have been delivered
                    # But we still allow emergency depot returns for infeasible instances
                    if current_load != 0:
                        # This should rarely happen now with improved action masking
                        pass

                    # Reset for next tour
                    current_time = 0
                    current_load = 0
                    picked_up_in_tour = set()
                else:
                    # Check time window constraint
                    time_window = td["time_windows"][b, next_node].item()  # time_windows includes depot at index 0
                    if current_time > time_window:
                        warnings.warn(
                            f"Time window violated at node {next_node}: arrival {current_time} > deadline {time_window}",
                            RuntimeWarning,
                        )

                    # Update load and check capacity
                    demand = td["demand"][b, next_node].item()  # demand includes depot at index 0

                    # Check if pickup or dropoff (interleaved: depot=0, odd=pickup, even=dropoff)
                    if next_node % 2 == 1:  # Pickup
                        current_load += demand
                        capacity = td["capacity"][b, 0].item()  # Assuming same capacity for all vehicles
                        if current_load > capacity:
                            warnings.warn(
                                f"Capacity exceeded at pickup {next_node}: load {current_load} > capacity {capacity}",
                                RuntimeWarning,
                            )
                        picked_up_in_tour.add(next_node)
                    else:  # Dropoff
                        # Check corresponding pickup was done by same vehicle
                        corresponding_pickup = next_node - 1
                        if corresponding_pickup not in picked_up_in_tour:
                            warnings.warn(
                                f"Dropoff {next_node} before pickup {corresponding_pickup} in same vehicle tour",
                                RuntimeWarning,
                            )
                        current_load += demand  # demand is negative for dropoff
                        if current_load < 0:
                            warnings.warn(
                                f"Negative load at dropoff {next_node}: load {current_load}",
                                RuntimeWarning,
                            )

                current_node = next_node
    
    def _make_spec(self, generator: DARPGenerator):
        """Make the observation and action specs from the parameters"""
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(shape=(1,), dtype=torch.int64),
            current_agent=Unbounded(shape=(1,), dtype=torch.int64),
            current_time=Unbounded(shape=(1,), dtype=torch.int64),
            current_load=Unbounded(shape=(1,), dtype=torch.int64),
            visited=Unbounded(shape=(generator.num_loc + 1,), dtype=torch.bool),
            picked_up_by_current=Unbounded(shape=(generator.num_loc + 1,), dtype=torch.bool),
            vehicle_finished=Unbounded(shape=(generator.num_agents,), dtype=torch.bool),
            i=Unbounded(shape=(1,), dtype=torch.int64),
            action_mask=Unbounded(shape=(generator.num_loc + 1,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)
    
    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        """Render the environment"""
        return render(td, actions, ax)
