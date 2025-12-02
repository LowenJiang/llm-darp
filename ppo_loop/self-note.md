This is note to myself about coding the DARPEnv. 
Why do you need a DARPEnv? Because the action now allows perturbation of request time windows. 
A disambiguation for myself: graph embedding, attention, etc are used in the PPO.py. the DVRPEnv remains a rather simple mechanism: 

First, lets go back to the PDPTWEnv. 
Basic usage: 
PDPTWEnv.reset(batch_size=[1]) -> 
Return: TensorDict(
    fields={
        action_mask: Tensor(shape=torch.Size([1, 61]), device=cpu, dtype=torch.bool, is_shared=False),
        capacity: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
        current_node: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        current_time: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        demand: Tensor(shape=torch.Size([1, 61]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        flexibility: Tensor(shape=torch.Size([1, 61]), device=cpu, dtype=torch.float32, is_shared=False),
        h3_indices: Tensor(shape=torch.Size([1, 61]), device=cpu, dtype=torch.int64, is_shared=False),
        i: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        locs: Tensor(shape=torch.Size([1, 61, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        pending_count: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        pending_schedule: Tensor(shape=torch.Size([1, 8]), device=cpu, dtype=torch.int64, is_shared=False),
        previous_action: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        time_windows: Tensor(shape=torch.Size([1, 61, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        travel_time_matrix: Tensor(shape=torch.Size([1, 31, 31]), device=cpu, dtype=torch.float32, is_shared=False),
        trip_metadata: NonTensorStack(
            [{1: {'flexibility': 'flexible for late dropoff, b...,
            batch_size=torch.Size([1]),
            device=None),
        used_capacity: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        user_id: Tensor(shape=torch.Size([1, 61]), device=cpu, dtype=torch.int64, is_shared=False),
        vehicle_capacity: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        visited: Tensor(shape=torch.Size([1, 61]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False)

This tensordict is rather important because it can be directly put to solver: 
out = self.policy(
    batched_td,
    phase='test',
    decode_type="greedy",
    return_actions=True
)

Note that for the terms with shape [1,61,]: these are growable (namely, when reset, we only have the depot and first two nodes; then each step, the shape grows by two, as two requests go into the state). 

Say we start with [batch_size,3]; and a new request come in. the reward is defined as the sliced [batch_size, 5] (with the time_windows field of 4 and 5 modified; namely fixed) and sliced [batch_size, 3] 's routing cost differnece. old - new. (so its likely to be negative). Note that for the non-61 fields, the oracle accepts varied size; so here is implied that each step, it slices two more nodes from the initial td and feeds into oracle. 

How does the steps work? for [batch_size, 61], we would have 30 decision steps. At i th step, the observable space is old/fixed requests [batch_size, 2i-1]and new request  [batch_size, 2i-1:2i+1] (fields that don't satisfy the [1,61, ()])shape (such as [1,1] or [travel_time_matrix] would stay constant; Actually, I might as well specify the fields that are dynamic in the dvrp: h3_indices, locs, time_windows) (The rest would just take on their initial value) 

The better way to see it is that the dvrp and pdptwenv has the same observation space; 
One Difference is that action for dvrp is defined as 16 perturbation actions that add on a request's pickup and dropoff time. 
ACTION_SPACE_MAP = [
    (-30, 0), (-30, 10), (-30, 20), (-30, 30),
    (-20, 0), (-20, 10), (-20, 20), (-20, 30),
    (-10, 0), (-10, 10), (-10, 20), (-10, 30),
    (0, 0), (0, 10), (0, 20), (0, 30),
]
Another difference is that reward for dvrp's ith step is sliced(batch_size, 0: 2i-1) - sliced(batch_size, 0:2i-1;perturbed(2i-1,2i+1))
Third difference being: at the jth step, it returns the j-1 request's h3_indices, time_windows and jth request's h3_indices and time_windows, as the following dict format: 
fixed: 
[h3_indices_pickup, time_window_00, time_window_10, h3_indices_dropoff, time_window_01, time_window_11]
......

new: 
[h3_indices_pickup, time_window_00, time_window_10, h3_indices_dropoff, time_window_01, time_window_11]

distance_matrix: 
from the pdptwenv's distance matrix

This will be exposed to PPO agent's observation and subsequent embedding.
