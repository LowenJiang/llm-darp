from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform # Here is for the random distributions

from inner_loop.rl4co.envs.common.utils import Generator, get_sampler
from inner_loop.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class PDPTWGenerator(Generator):
    """
    Data Generator for pickup and dropoff problem with time window constraint. 
    Args: 
        - 1 depot
        - number of locations sans the depot
            - 'num_loc' / 2 pickup locations
            - 'num_loc' / 2 deliveryt loctions 
        - min_loc: minimum value for the location coordinates
        - max_loc: maximum value for the location coordinates
        - vehicle_speed: controls the ratio between location and time
        - time_window: duration of time window, by defualt 30min
        
    Generation methods
    Goal: 
    Generate a PDPTW instance that is guaranteed to be solvable by 1 vehicle. 
    In PDPTW, each request will have a pickup and dropoff node, each has a demand (positive for pickup, negative for dropoff) and a 30min time window
    Solvable means, it is possible for 1 vehicle's one-day operation to fulfill every request in the said time windows.

    Method: Because randomly sampling nodes and time-windows will not guarantee feaxibility, we reverse-sample a vehicle trajectory. 
    We first start from the real min-resolution time, and at last we transform it to 30min time windows. 

    Desired output for PDPTWGenerator: 
    Attribute of PDPTW instance:  _StringKeys(dict_keys(['locs', 'depot', 'time_windows', 'demand', 'capacity']))
    depot: torch.Size([batch_size, 2])
    locs: torch.Size([batch_size, num_loc, 2]) 
    capacity: torch.Size([batch_size]) # Vehicle capacity   
    time_windows: torch.Size([batch_size, num_loc, 2]) # Each loc has (for example) 60, 90 (say, in 30min interval)
    demand: torch.Size([batch_size, locs])  # positive / negative for pickup / dropoff nodes

    1. Randomly sample num_loc + 1 (incl output) nodes in the xy plane of (min_loc, max_loc; min_loc, max_loc)
    2. Generate a random hamilton cycle going from depot, traversing all nodes and back to depot. 
    3. Set travel time: 
        3.1 The time-length connecting the two nodes in the said trajectory is the 2-norm / vehicle_speed (unit: minutes)
            #! Note: This travel duration gives the minimum "actual travel time" between two nodes on the trajectory.
        3.2 Consider the "daily operation time": 6:00-22:00, in terms of minutes: 360 to 1320 min; duration = 960min
        3.3 Consider the waiting time to be distributed towards the nodes: free-time =  960 - (sum(hamilton_cycle_travel_time))
        3.4 Randomly assign free-time added to all nodes: node-i has free-time-i. sum_i(free-time-i) = free-time
            #! You should Figure out how to randomly allocate the free time to every nodes
            for node-i in iter(num_loc): # Here the output is not included and counted from 0.
                if i > 0: node-actual-travel-time(i) = node-actual-travel-time(i-1) + hamilton-travel-time(i-1,i) + free-time-i
                if i = 0: node-actual-travel-time(i) = 360 + free-time-i
                
    4 Assign pickup - dropoff labels to the nodes
        4.1 Instantiate a pickup - dropoff - demand table of size [num_loc/2, 3], filled with 0
        4.2 Starting from depot and iterate sequentially through all nodes in the hamilton cycle. Set remaining-capacity=5 to begin: 
            If: len(column(0)==0) >= len(column(1)==0) # Vehicle empty at this time
                next node must be pickup. Set the first 0 of column 0 to the node-actual-travel-time (min)
                Randomly select demand from {1,2,3} with {60%, 20%, 20%} probability; remaining-capacity -= demand
                set the row=pickup, col=3 to demand.
            Elif len(column(0)==0) < len(column(1)==0):
                if remaining-capacity > 0: 
                    next node randomly(with 0.5 bernouli) select between pickup and dropoff
                    if next node is selected pickup, 
                        set first 0 of column 0 to time window. 
                        pickup-demand = randomly sample {1,2...remaining-capacity} with 1=60%; 2,3...remaining-capacity summed to 40%
                        remaining-capacity -= pickup-demand
                        [row-changed, 3] = pickup-demand
                    if next node is selected pickup, 
                        set first 0 of column 0 to time window. 
                        remaining-capacity += [row-changed, 3]

            Repeat untill all 0's have been replaced with actual travel time. 
        4.3 Define pickup-dropoff pairs: If node i is assigned to table(row-x, 0); node j is assinged to table(row-x, 0);
            then node i and node j are a pickup-dropoff pair. 

        
    5. Structuralize to output: 
        depot: depot-location
        locs: node locations: Alternate between pickup and dropoff pairs. For example, first and second entry is the first pickup and first dropoff.
        capacity: vehicel capacity, default to 5 and set in init to PDPTWGenerator
        time_windows: In the same order as locs wrt pickup and dropoff; each node-entry is the 30min interval start and finish time.
            for example, if a node's actual travel time is 670, then time_windows for it is (660, 690)
        demand: In the same order as locs wrt pickup and dropoff; positive for pickup and negative to dropoff




        
        


    """
    def __init__(
        self,
        num_loc: int = 40,
        min_loc: float = 0.0,
        max_loc: float = 100.0,
        vehicle_speed: float = 5.0, # Consider the avg travel time to be 20min
        vehicle_capacity: int = 5,
        time_window_width: float = 30.0, # minutes
        min_time: float = 360.0, # 6:00 AM in minutes
        max_time: float = 1320.0, # 10:00 PM in minutes
        service_duration: float = 10, # Service time at each location in minutes
        **kwargs
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.vehicle_speed = vehicle_speed
        self.vehicle_capacity = vehicle_capacity
        self.time_window_width = time_window_width
        self.min_time = min_time
        self.max_time = max_time
        self.service_duration = service_duration
        super().__init__(**kwargs)

    def _generate(self, batch_size) -> TensorDict:
        """
        Generate PDPTW instances by reverse-sampling feasible vehicle trajectories.

        Args:
            batch_size: Number of instances to generate (can be int, list, or torch.Size)

        Returns:
            TensorDict with keys: depot, locs, capacity, time_windows, demand
        """
        # Handle batch_size as list, torch.Size, or int
        if isinstance(batch_size, (list, tuple)):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1
        elif isinstance(batch_size, torch.Size):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1

        # Ensure batch_size is an int
        if not isinstance(batch_size, int):
            batch_size = int(batch_size)

        # Ensure batch_size is at least 1
        if batch_size <= 0:
            log.warning(f"Invalid batch_size {batch_size}, setting to 1")
            batch_size = 1

        # Step 1: Randomly sample num_loc + 1 nodes (depot + locations)
        # Depot location
        depot = torch.rand(batch_size, 2) * (self.max_loc - self.min_loc) + self.min_loc

        # All node locations (including those that will become the trajectory)
        all_locs = torch.rand(batch_size, self.num_loc, 2) * (self.max_loc - self.min_loc) + self.min_loc

        # Step 2: Generate random Hamilton cycle
        # Create random permutation for each batch
        hamilton_cycles = torch.stack([torch.randperm(self.num_loc) for _ in range(batch_size)])

        # Step 3: Calculate travel times and allocate free time
        actual_travel_times = self._calculate_travel_times(depot, all_locs, hamilton_cycles, batch_size)

        # Step 4: Assign pickup-dropoff pairs with demand
        pickup_dropoff_table, node_types = self._assign_pickup_dropoff(actual_travel_times, batch_size)

        # Step 5: Structure output
        td = self._structure_output(depot, all_locs, pickup_dropoff_table, node_types,
                                     actual_travel_times, batch_size)

        # Step 6: Randomize pair order
        td = self._randomize_pairs(td)

        return td

    def _calculate_travel_times(self, depot, all_locs, hamilton_cycles, batch_size):
        """
        Calculate actual travel times for each node in the Hamilton cycle.

        Returns:
            actual_travel_times: [batch_size, num_loc] tensor with actual arrival times
        """
        actual_travel_times = torch.zeros(batch_size, self.num_loc)

        for b in range(batch_size):
            # Get the Hamilton cycle for this batch
            cycle = hamilton_cycles[b]

            # Calculate travel durations between consecutive nodes in the cycle
            travel_durations = torch.zeros(self.num_loc)

            for i in range(self.num_loc):
                if i == 0:
                    # Travel from depot to first node
                    prev_loc = depot[b]
                    curr_loc = all_locs[b, cycle[i]]
                else:
                    # Travel from previous node to current node
                    prev_loc = all_locs[b, cycle[i-1]]
                    curr_loc = all_locs[b, cycle[i]]

                # Calculate Euclidean distance and convert to time
                distance = torch.norm(curr_loc - prev_loc)
                travel_durations[i] = distance / self.vehicle_speed

            # Calculate total travel time for the cycle (including return to depot)
            last_loc = all_locs[b, cycle[-1]]
            return_distance = torch.norm(depot[b] - last_loc)
            total_cycle_time = travel_durations.sum() + return_distance / self.vehicle_speed

            # Calculate free time available
            operation_duration = self.max_time - self.min_time
            free_time = operation_duration - total_cycle_time

            if free_time < 0:
                log.warning(f"Negative free time in batch {b}: {free_time}. Adjusting to 0.")
                free_time = torch.tensor(0.0)

            # Randomly allocate free time to nodes using Dirichlet-like distribution
            # Use exponential distribution and normalize
            free_time_allocation = torch.rand(self.num_loc)
            free_time_allocation = free_time_allocation / free_time_allocation.sum() * free_time

            # Calculate actual travel times
            for i in range(self.num_loc):
                if i == 0:
                    actual_travel_times[b, cycle[i]] = self.min_time + travel_durations[i] + free_time_allocation[i]
                else:
                    actual_travel_times[b, cycle[i]] = (
                        actual_travel_times[b, cycle[i-1]] + travel_durations[i] + free_time_allocation[i]
                    )

        return actual_travel_times

    def _assign_pickup_dropoff(self, actual_travel_times, batch_size):
        """
        Assign pickup and dropoff labels to nodes in the Hamilton cycle.

        Returns:
            pickup_dropoff_table: [batch_size, num_loc//2, 3] with (pickup_time, dropoff_time, demand)
            node_types: [batch_size, num_loc] with 0 for pickup, 1 for dropoff
        """
        num_pairs = self.num_loc // 2
        pickup_dropoff_table = torch.zeros(batch_size, num_pairs, 3)
        node_types = torch.zeros(batch_size, self.num_loc, dtype=torch.long)

        for b in range(batch_size):
            # Sort nodes by actual travel time to process in order
            sorted_indices = torch.argsort(actual_travel_times[b])

            remaining_capacity = self.vehicle_capacity
            pickup_count = 0
            dropoff_count = 0
            pair_idx = 0
            open_pickups = []  # Track which pairs still need dropoff

            for node_idx in sorted_indices:
                node_time = actual_travel_times[b, node_idx].item()

                # Decide if this is pickup or dropoff
                if pickup_count == dropoff_count:  # Vehicle empty
                    # Must be pickup
                    is_pickup = True
                elif pickup_count >= num_pairs:  # All pickups assigned
                    # Must be dropoff
                    is_pickup = False
                elif dropoff_count >= num_pairs:  # All dropoffs assigned (shouldn't happen)
                    # Must be pickup
                    is_pickup = True
                elif remaining_capacity <= 0:  # Vehicle full
                    # Must be dropoff
                    is_pickup = False
                else:
                    # Random choice with 0.5 probability
                    is_pickup = torch.rand(1).item() > 0.5

                if is_pickup:
                    # Assign as pickup node
                    node_types[b, node_idx] = 0  # 0 for pickup

                    # Sample demand
                    rand = torch.rand(1).item()
                    if rand < 0.8 or remaining_capacity == 1:
                        demand = 1
                    else:
                        # Sample from 2 to remaining_capacity
                        max_demand = int(min(remaining_capacity, 3))
                        demand = torch.randint(2, max_demand + 1, (1,)).item()

                    # Store in table
                    pickup_dropoff_table[b, pair_idx, 0] = node_time
                    pickup_dropoff_table[b, pair_idx, 2] = demand

                    remaining_capacity -= demand
                    open_pickups.append(pair_idx)
                    pair_idx += 1
                    pickup_count += 1
                else:
                    # Assign as dropoff node
                    node_types[b, node_idx] = 1  # 1 for dropoff

                    # Match with an open pickup
                    if open_pickups:
                        matched_pair = open_pickups.pop(0)
                        pickup_dropoff_table[b, matched_pair, 1] = node_time
                        demand = pickup_dropoff_table[b, matched_pair, 2].item()
                        remaining_capacity += demand
                        dropoff_count += 1

        return pickup_dropoff_table, node_types

    def _structure_output(self, depot, all_locs, pickup_dropoff_table, node_types,
                          actual_travel_times, batch_size):
        """
        Structure the output into the required TensorDict format.

        Returns:
            TensorDict with keys: depot, locs, capacity, time_windows, demand
        """
        num_pairs = self.num_loc // 2

        # Initialize output tensors
        locs = torch.zeros(batch_size, self.num_loc, 2)
        time_windows = torch.zeros(batch_size, self.num_loc, 2)
        demand = torch.zeros(batch_size, self.num_loc)
        capacity = torch.full((batch_size,), self.vehicle_capacity, dtype=torch.float)
        durations = torch.full((batch_size, self.num_loc), self.service_duration, dtype=torch.float)

        for b in range(batch_size):
            # Create mapping from original indices to output indices
            # Output format: alternate pickup-dropoff pairs
            output_idx = 0

            for pair_idx in range(num_pairs):
                pickup_time = pickup_dropoff_table[b, pair_idx, 0].item()
                dropoff_time = pickup_dropoff_table[b, pair_idx, 1].item()
                pair_demand = pickup_dropoff_table[b, pair_idx, 2].item()

                # Find which nodes correspond to this pickup and dropoff
                pickup_node = None
                dropoff_node = None

                for node_idx in range(self.num_loc):
                    node_time = actual_travel_times[b, node_idx].item()

                    if abs(node_time - pickup_time) < 0.01 and node_types[b, node_idx] == 0:
                        if pickup_node is None:
                            pickup_node = node_idx
                            node_types[b, node_idx] = -1  # Mark as used

                    if abs(node_time - dropoff_time) < 0.01 and node_types[b, node_idx] == 1:
                        if dropoff_node is None:
                            dropoff_node = node_idx
                            node_types[b, node_idx] = -1  # Mark as used

                # Assign pickup node (even index)
                if pickup_node is not None:
                    locs[b, output_idx] = all_locs[b, pickup_node]

                    # Create 30-min time window
                    tw_start = (pickup_time // self.time_window_width) * self.time_window_width
                    tw_end = tw_start + self.time_window_width
                    time_windows[b, output_idx, 0] = tw_start
                    time_windows[b, output_idx, 1] = tw_end

                    demand[b, output_idx] = pair_demand  # Positive for pickup
                output_idx += 1

                # Assign dropoff node (odd index)
                if dropoff_node is not None:
                    locs[b, output_idx] = all_locs[b, dropoff_node]

                    # Create 30-min time window
                    tw_start = (dropoff_time // self.time_window_width) * self.time_window_width
                    tw_end = tw_start + self.time_window_width
                    time_windows[b, output_idx, 0] = tw_start
                    time_windows[b, output_idx, 1] = tw_end

                    demand[b, output_idx] = -pair_demand  # Negative for dropoff
                output_idx += 1

        # Prepend depot information to durations and time_windows
        # Depot has 0 service duration and time window covering full operating hours
        depot_durations = torch.zeros(batch_size, 1, dtype=torch.float)
        depot_time_windows = torch.zeros(batch_size, 1, 2, dtype=torch.float)
        depot_time_windows[:, 0, 0] = self.min_time
        depot_time_windows[:, 0, 1] = self.max_time

        durations_with_depot = torch.cat([depot_durations, durations], dim=1)
        time_windows_with_depot = torch.cat([depot_time_windows, time_windows], dim=1)

        return TensorDict({
            "depot": depot,
            "locs": locs,
            "capacity": capacity,
            "time_windows": time_windows_with_depot,
            "demand": demand,
            "durations": durations_with_depot
        }, batch_size=[batch_size])

    def _randomize_pairs(self, td: TensorDict) -> TensorDict:
        """
        Randomize the order of pickup-dropoff pairs while preserving each pair's structure.

        Args:
            td: TensorDict with locs, time_windows (including depot), demand, durations (including depot)

        Returns:
            TensorDict with randomized pair order
        """
        batch_size = td.batch_size[0] if isinstance(td.batch_size, tuple) else td.batch_size
        num_pairs = self.num_loc // 2

        # Create new tensors for randomized output
        # locs and demand don't include depot
        randomized_locs = torch.zeros_like(td['locs'])
        randomized_demand = torch.zeros_like(td['demand'])

        # time_windows and durations include depot at index 0
        randomized_time_windows = torch.zeros_like(td['time_windows'])
        randomized_durations = torch.zeros_like(td['durations'])

        # Copy depot information (index 0) - unchanged
        randomized_time_windows[:, 0] = td['time_windows'][:, 0]
        randomized_durations[:, 0] = td['durations'][:, 0]

        for b in range(batch_size):
            # Generate random permutation of pair indices
            pair_permutation = torch.randperm(num_pairs)

            # Reorder pairs according to permutation
            for new_pair_idx, old_pair_idx in enumerate(pair_permutation):
                # Original indices for this pair (in locs/demand arrays without depot)
                old_pickup_idx = old_pair_idx * 2
                old_dropoff_idx = old_pair_idx * 2 + 1

                # New indices for this pair
                new_pickup_idx = new_pair_idx * 2
                new_dropoff_idx = new_pair_idx * 2 + 1

                # Copy pickup data
                randomized_locs[b, new_pickup_idx] = td['locs'][b, old_pickup_idx]
                randomized_demand[b, new_pickup_idx] = td['demand'][b, old_pickup_idx]
                # time_windows and durations indices are offset by 1 (depot at index 0)
                randomized_time_windows[b, new_pickup_idx + 1] = td['time_windows'][b, old_pickup_idx + 1]
                randomized_durations[b, new_pickup_idx + 1] = td['durations'][b, old_pickup_idx + 1]

                # Copy dropoff data
                randomized_locs[b, new_dropoff_idx] = td['locs'][b, old_dropoff_idx]
                randomized_demand[b, new_dropoff_idx] = td['demand'][b, old_dropoff_idx]
                # time_windows and durations indices are offset by 1 (depot at index 0)
                randomized_time_windows[b, new_dropoff_idx + 1] = td['time_windows'][b, old_dropoff_idx + 1]
                randomized_durations[b, new_dropoff_idx + 1] = td['durations'][b, old_dropoff_idx + 1]

        # Return new TensorDict with randomized data
        return TensorDict({
            "depot": td['depot'],
            "locs": randomized_locs,
            "capacity": td['capacity'],
            "time_windows": randomized_time_windows,
            "demand": randomized_demand,
            "durations": randomized_durations
        }, batch_size=[batch_size])
