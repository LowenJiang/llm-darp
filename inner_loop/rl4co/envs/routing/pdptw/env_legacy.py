from typing import Optional
import warnings

import torch

from tensordict.tensordict import TensorDict

from torchrl.data import Unbounded, Bounded, Composite

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.cvrptw.env import CVRPTWEnv
from rl4co.utils.ops import gather_by_index, get_distance, get_geo_distance, get_geod_tour_length

from .generator import PDPTWGenerator
from .sf_generator import SFGenerator
from .render import render


class PDPTWEnv(CVRPTWEnv):
    """
    Pickup and Delivery Problem with Time Windows (PDPTW) environment.

    Inherits from CVRPTWEnv and adds precedence constraints:
    - Each pickup must be visited before its corresponding delivery
    - Pickup-delivery pairs are consecutive indices: (1,2), (3,4), (5,6), etc.
      where odd indices are pickups and even indices are deliveries (after depot at index 0)

    The environment is made of num_loc + 1 locations:
        - 1 depot (index 0)
        - num_loc locations consisting of num_loc/2 pickup-delivery pairs

    Observations:
        - location of depot, pickup and delivery nodes
        - demand of each node (positive for pickup, negative for delivery)
        - service duration of each pickup and delivery node
        - time window of each location
        - current location of the vehicle
        - current capacity of the vehicle
        - current time
        - visited locations
        - current step counter
        - cumulative depot penalty
        - unfulfilled requests counter
        - cumulative unfulfilled penalty

    Constraints:
        - The tour starts and ends at the depot
        - Each pickup location must be visited before its corresponding delivery location
        - The vehicle cannot visit the same location twice
        - The vehicle cannot exceed capacity constraints
        - The vehicle must start service within the time window of each location

    Finish Condition:
        - The vehicle has visited all pickup and delivery nodes and returned to depot

    Reward:
        - negative tour length minus the accumulated depot penalties and unfulfilled penalties

    Args:
        generator: PDPTWGenerator instance as the data generator
        generator_params: parameters for the generator
        depot_penalty: penalty applied each time the depot is revisited before finishing the route
        unfulfilled_penalty: penalty applied for each delivery completed after its time window
    """

    name = "pdptw"

    def __init__(
        self,
        generator: SFGenerator = None,
        generator_params: dict = {},
        depot_penalty: int = 50, #Penalty for each additioanl vehicle
        fleet_count: int = 5,     #maximum number of vehicles used
        vehicle_speed: float = 0.3,
        unfulfilled_penalty: float = 100.0,
        **kwargs,
    ):
        # Set attributes BEFORE super().__init__() so they're available in _make_spec
        self.depot_penalty_factor = int(depot_penalty)
        self.fleet_count = fleet_count
        self.vehicle_speed = vehicle_speed
        self.unfulfilled_penalty = unfulfilled_penalty
        # Create generator BEFORE super().__init__()
        if generator is None:
            generator = SFGenerator(**generator_params)

        # Parent classes don't forward generator properly, so let parent init run,
        # then override generator and rebuild specs
        super().__init__(**kwargs)

        # Override with our generator (parent set wrong one)
        self.generator = generator

        # Rebuild specs with correct generator
        self._make_spec(self.generator)

        # Ensure backward compatibility
        self._ensure_depot_penalty_attr(default=depot_penalty)

    # Just for backwards compatibility for when the ckpt doesn't take penalty terms
    def _ensure_depot_penalty_attr(self, default: int | None = None):
        """Ensure backward compatibility for checkpoints missing depot_penalty_cost."""
        if not hasattr(self, "depot_penalty_cost"):
            self.depot_penalty_factor = int(100 if default is None else default)

    def _make_spec(self, generator: PDPTWGenerator):
        """Extend parent specs with step counter and vehicle speed."""
        # Check if we have the right generator type with required attributes
        if not hasattr(generator, 'num_customers'):
            # Parent is calling with wrong generator type, just let parent set its specs
            super()._make_spec(generator)
            return

        #super()._make_spec(generator)
        # From generator information - use the generator parameter, not self.generator
        locs = Bounded(low=-180, high=180, shape = (generator.num_customers*2+1, 2), dtype=torch.float32)
        time_windows = Bounded(low=0, high=1440, shape=(generator.num_customers*2+1, 2), dtype=torch.int64) # Suppose integer min
        demand = Bounded(low=-generator.vehicle_capacity, high=generator.vehicle_capacity,
                         shape=(generator.num_customers*2,), dtype=torch.int64)

        # Current vehicle
        current_node = Unbounded(shape=(1), dtype=torch.int64)
        current_loc = Bounded(low=-180, high=180, shape=(2), dtype=torch.float32)
        current_time = Bounded(low=0, high=1440, shape=(1), dtype=torch.float32) # Using float32 for fractional travel times

        i = Unbounded(shape=(1), dtype=torch.int64) # num of locations visited
        #vehicles_used = Unbounded(shape=(1), dtype=torch.int64) # Penalzie over fleet_count vehicles

        action_mask = Bounded(low=0, high=1, shape=(generator.num_customers*2+1, 1), dtype=torch.bool) #incl depot
        #feasibility = self.Bounded(shape=(1), dtype=torch.bool) #store if over fleet_count vehicles are used
        self.observation_spec = Composite(
            locs=locs,
            time_windows=time_windows,
            demand=demand,
            current_node=current_node,
            current_loc=current_loc,
            current_time=current_time,
            i=i,
            action_mask=action_mask
        )

        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.generator.num_customers * 2 + 1,  # Exclusive upper bound: need +1 to include depot (0) and all customers (1 to 2*num_customers)
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)


    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device
        # Prepare durations: add depot with zero duration at the beginning
        durations_with_depot = torch.cat(
            (torch.zeros(*batch_size, 1, device=device, dtype=td["demand"].dtype),
             td.get("durations", torch.zeros_like(td["demand"]))),
            dim=-1
        )

        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),
                "demand": td["demand"],
                "durations": durations_with_depot,  # Service durations (including depot)
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=device,
                ),
                "time_windows": td["time_windows"],
                "feasibility": torch.ones(
                    *batch_size, 1, dtype=torch.bool, device=device
                ),
                "vehicles_used": torch.zeros( #!Update when returned to depot
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.generator.vehicle_capacity, device=device
                ),
                "vehicle_penalty": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "unfulfilled_requests": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "unfulfilled_penalty": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "vehicle_speed": torch.full(
                    (*batch_size, 1), self.vehicle_speed, device=device
                ),
                "i": torch.zeros(
                    *batch_size, 1, dtype=torch.int64, device=device
                )
            },
            batch_size=batch_size,
        )

        td_reset.set("action_mask", self.get_action_mask(td_reset))

        return td_reset

    # Update the dynamics wrt vehicles_used and vehicle_penalty !
    def _step(self, td: TensorDict) -> TensorDict:

        batch_size = td["locs"].shape[0]
        prev_time = td["current_time"]

        # Compute travel time from current location to action location
        current_loc = gather_by_index(td["locs"], td["current_node"])
        action_loc = gather_by_index(td["locs"], td["action"])
        dist = get_geo_distance(current_loc, action_loc)

        # Ensure dist has shape [batch_size, 1] to avoid broadcasting issues
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)

        vehicle_speed = td.get("vehicle_speed", torch.tensor(0.3, device=dist.device))
        if vehicle_speed.dim() == 0:
            vehicle_speed = vehicle_speed.unsqueeze(0).unsqueeze(0)
        while vehicle_speed.dim() < dist.dim():
            vehicle_speed = vehicle_speed.unsqueeze(-1)
        travel_time = (dist / vehicle_speed).reshape([batch_size, 1])
        node_time_windows = gather_by_index(td["time_windows"], td["action"])
        start_times = node_time_windows[..., 0].reshape([batch_size, 1]).to(prev_time.dtype)
        end_times = node_time_windows[..., 1].reshape([batch_size, 1]).to(prev_time.dtype)
        service_start = torch.max(prev_time + travel_time, start_times)
        
        # Track unfulfilled deliveries (service starts after time window ends)
        is_delivery = ((td["action"] > 0) & (td["action"] % 2 == 0)).unsqueeze(-1)
        is_late = (service_start > end_times).to(torch.float32)
        unfulfilled_mask = is_delivery.to(torch.float32) * is_late
        
        td["unfulfilled_requests"] = td["unfulfilled_requests"] + unfulfilled_mask.to(td["unfulfilled_requests"].dtype)
        td["unfulfilled_penalty"] = td["unfulfilled_penalty"] + unfulfilled_mask * self.unfulfilled_penalty

        td = super()._step(td)

        done = td["done"].bool()
        current_node = td["current_node"]
        while done.dim() < current_node.dim():
            done = done.unsqueeze(-1)

        returned_to_depot = current_node == 0 # Does the starting condition at node 0 count?
        new_vehicle_mask = returned_to_depot & (~done)

        td["vehicles_used"] = td["vehicles_used"] + new_vehicle_mask.to(td["vehicles_used"].dtype) * 1

        oversubscribed_mask = new_vehicle_mask & (td["vehicles_used"] > self.fleet_count)
        td["feasibility"] = td["feasibility"] & (~oversubscribed_mask) #mask to infeasible
        td["vehicle_penalty"] = td["vehicle_penalty"] + oversubscribed_mask.to(td["vehicle_penalty"].dtype) * self.depot_penalty_factor
        td["i"] = td["i"] + 1
        return td

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:

        locs_ordered = torch.cat(
            [
                td["locs"][..., 0:1, :],  # depot
                gather_by_index(td["locs"], actions),  # order locations
            ],
            dim=1,
        )
        base_reward = -get_geod_tour_length(locs_ordered)
        # Use the penalty already accumulated in td["vehicle_penalty"] during rollout
        depot_penalty = td["vehicle_penalty"].squeeze(-1) 
        unfulfilled_penalty = td["unfulfilled_penalty"].squeeze(-1)
        return base_reward - depot_penalty - unfulfilled_penalty

    def __setstate__(self, state):
        super().__setstate__(state)
        self._ensure_depot_penalty_attr()

    def __getattr__(self, name):
        """Handle missing depot_penalty_cost for backward compatibility."""
        if name == "depot_penalty_cost":
            # Default to 100 for backward compatibility
            self.depot_penalty_cost = 100
            return self.depot_penalty_cost
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    #! This need to be changed: introduce "unmaskable" nodes, and mask actions that would mask the unmaskables
    # NOte: On the duration
    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """
        Extend CVRPTW masking with PDPTW precedence constraints:
            - Deliveries become available only after the paired pickup was visited
            - Depot can only be revisited once all pickups and deliveries are completed
        """
        base_mask = CVRPEnv.get_action_mask(td)
        current_loc = gather_by_index(td["locs"], td["current_node"])
        dist = get_geo_distance(current_loc[..., None, :], td["locs"])
        vehicle_speed = td.get("vehicle_speed", None)
        if vehicle_speed is None:
            speed = torch.ones_like(dist[..., :1], device=dist.device) * 5
        else:
            speed = vehicle_speed.to(dist.device)
            while speed.dim() < dist.dim():
                speed = speed.unsqueeze(-1)
        travel_time = dist / speed
        td.update({"current_loc": current_loc, "distances": travel_time})
        can_reach_in_time = td["current_time"] + travel_time <= td["time_windows"][..., 1]
        mask = base_mask & can_reach_in_time
        visited = td["visited"].bool()

        # Enforce pickup-before-delivery constraint (indices: pickup 1,3,5..., delivery 2,4,6...)
        pickups_visited = visited[..., 1::2]
        deliveries_visited = visited[..., 2::2]
        mask[..., 2::2] &= pickups_visited

        # Deliveries whose pickups are done become unmaskable and ignore time windows
        unmaskable_deliveries = pickups_visited & (~deliveries_visited)
        mask[..., 2::2] |= base_mask[..., 2::2] & unmaskable_deliveries
        """
        
        # NEW SETTING
        # Depot is available when there are no pending deliveries or all customers are done
        all_customers_done = visited[..., 1:].all(dim=-1, keepdim=True)
        no_pending_deliveries = ~unmaskable_deliveries.any(dim=-1, keepdim=True)
        mask[..., :1] &= (all_customers_done | no_pending_deliveries)

        # Emergency escape: if no valid customer actions, depot must be available
        has_customer_actions = mask[..., 1:].any(dim=-1, keepdim=True)
        mask[..., :1] |= ~has_customer_actions

        """
        # ORIGINAL SETTING
        # Depot is only available when all customers are done, unless no customer actions exist
        all_customers_done = visited[..., 1:].all(dim=-1, keepdim=True)
        mask[..., :1] &= all_customers_done
        has_customer_actions = mask[..., 1:].any(dim=-1, keepdim=True)
        mask[..., :1] |= ~has_customer_actions

        return mask

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """
        Check that a solution satisfies all PDPTW constraints.
        """
        # First check parent constraints (capacity, time windows, etc.)
        CVRPTWEnv.check_solution_validity(td, actions)

        batch_size = actions.shape[0]

        # Check precedence constraints: each pickup must come before its delivery
        num_locs = td["locs"].shape[1]  # Includes depot
        num_pairs = (num_locs - 1) // 2

        for b in range(batch_size):
            action_list = actions[b].tolist()

            for pair_idx in range(num_pairs):
                pickup_idx = 2 * pair_idx + 1
                delivery_idx = 2 * pair_idx + 2

                # Find positions in action sequence
                try:
                    pickup_pos = action_list.index(pickup_idx)
                    delivery_pos = action_list.index(delivery_idx)

                    if pickup_pos >= delivery_pos:
                        warnings.warn(
                            f"Batch {b}, Pair {pair_idx}: Pickup at index {pickup_idx} "
                            f"(position {pickup_pos}) should precede delivery at "
                            f"index {delivery_idx} (position {delivery_pos}).",
                            RuntimeWarning,
                        )
                except ValueError:
                    warnings.warn(
                        f"Batch {b}, Pair {pair_idx}: Missing pickup or delivery in solution.",
                        RuntimeWarning,
                    )

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        """Render the PDPTW solution."""
        return render(td, actions, ax)