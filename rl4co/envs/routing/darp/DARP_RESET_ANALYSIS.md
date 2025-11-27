# Complete Analysis: DARPEnv Reset Process

## Overview
When you call `env.reset(batch_size=[3])` on a DARPEnv, a complex initialization process occurs that generates problem instances and initializes all the environment state needed for the routing task. This document traces the complete flow.

---

## Call Sequence

### 1. Initial Call
```python
env.reset(batch_size=[3])
```

### 2. RL4COEnvBase.reset() [base.py:135-143]
```python
def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
    """Reset function to call at the beginning of each episode"""
    if batch_size is None:
        batch_size = self.batch_size if td is None else td.batch_size
    if td is None or td.is_empty():
        td = self.generator(batch_size=batch_size)  # ← Call generator
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
    self.to(td.device)
    return super().reset(td, batch_size=batch_size)  # ← Call parent's reset
```

**What happens:**
- `batch_size` is normalized to `[3]` (list format)
- Since `td` is None, the generator is called: `self.generator(batch_size=[3])`
- The result is passed to TorchRL's `EnvBase.reset()` which calls `_reset()`

### 3. Generator Call [generator.py:25-27]
```python
# In Generator base class
def __call__(self, batch_size) -> TensorDict:
    batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
    return self._generate(batch_size)
```

**Flow:**
- Input: `batch_size = [3]`
- Calls `DARPGenerator._generate([3])`
- Returns initial TensorDict with problem data

---

## Data Generation: DARPGenerator._generate()
### File: `/Users/jiangwolin/Desktop/Research/llm-rl/rl4co/rl4co/envs/routing/darp/generator.py` (Lines 103-181)

### Parameters (from __init__)
```python
num_loc: int = 20              # Number of locations (must be even)
num_agents: int = 5            # Number of vehicles
min_loc: float = 0.0           # Min coordinate value
max_loc: float = 100.0         # Max coordinate value
vehicle_capacity: int = 5      # Capacity per vehicle
min_demand: int = 1            # Min pickup demand
max_demand: int = 3            # Max pickup demand
vehicle_speed: float = 25.0    # Speed (for time conversion)
min_pickup_tw: int = 10        # Min pickup time window
max_pickup_tw: int = 30        # Max pickup time window
```

### Generation Steps

#### Step 1: Sample Depot Location
```python
depot = self.depot_sampler.sample((*batch_size, 2))
# Result: [3, 2] - one depot location per batch element
# Example: [[47.3, 52.1], [61.8, 38.4], [29.5, 71.2]]
```

#### Step 2: Sample Node Locations
```python
locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
# Result: [3, 20, 2] - 20 nodes per batch element
# Includes both pickup and dropoff locations (interleaved)
```

#### Step 3: Sample and Create Demands
```python
# Sample base demand values for pickups only
demand_values = self.demand_sampler.sample((*batch_size, self.num_loc // 2))
# Result: [3, 10] - one value per pickup
demand_values = (demand_values.int() + 1).float()  # Convert to [min_demand, max_demand]

# Create full demand tensor with negative values for dropoffs
demand = torch.zeros(*batch_size, self.num_loc)
for i in range(self.num_loc // 2):
    demand[..., 2*i]     = demand_values[..., i]      # Pickup (positive)
    demand[..., 2*i + 1] = -demand_values[..., i]     # Dropoff (negative)

# Result: [3, 20] with pattern:
# Index:    0   1   2   3   4   5   ...
# Value:    0  +2  -2  +1  -1  +3  ...  (0=depot, odd=pickup, even=dropoff)
```

#### Step 4: Generate Time Windows
```python
# For each pickup-dropoff pair
time_windows = torch.zeros(*batch_size, self.num_loc, dtype=torch.long)

for i in range(self.num_loc // 2):
    pickup_idx = 2 * i
    dropoff_idx = 2 * i + 1
    
    # Sample pickup time window from [min_pickup_tw, max_pickup_tw]
    pickup_tw = torch.randint(min_pickup_tw, max_pickup_tw + 1, size=batch_size)
    # Result: [3] with values like [15, 22, 18]
    
    # Calculate distance between pickup and dropoff
    dist = torch.norm(locs[..., pickup_idx, :] - locs[..., dropoff_idx, :], p=2, dim=-1)
    travel_time = dist / vehicle_speed
    
    # Dropoff must be reachable from pickup
    # Sample dropoff TW in [pickup_tw + travel_time, pickup_tw + 2*travel_time]
    # (ensures vehicle has time to get from pickup to dropoff and back)
```

#### Step 5: Create Capacity Tensor
```python
capacity = torch.full((*batch_size, self.num_agents), self.vehicle_capacity, dtype=torch.long)
# Result: [3, 5] - all 5s (each vehicle has capacity 5)
```

#### Step 6: Return Generated Data
```python
return TensorDict(
    {
        "locs": locs,                    # [B, num_loc, 2]      - customer locations
        "depot": depot,                  # [B, 2]               - depot location
        "time_windows": time_windows,    # [B, num_loc]         - time window deadline for each node
        "demand": demand,                # [B, num_loc]         - positive=pickup, negative=dropoff
        "capacity": capacity,            # [B, num_agents]      - vehicle capacity
    },
    batch_size=batch_size,  # [3]
)
```

**Generated TensorDict Fields:**
| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `locs` | [3, 20, 2] | float32 | (x, y) coordinates for each node |
| `depot` | [3, 2] | float32 | Depot location |
| `demand` | [3, 20] | float32 | Demand: positive (pickup), negative (dropoff) |
| `time_windows` | [3, 20] | int64 | Time deadline for each node (1-48) |
| `capacity` | [3, 5] | int64 | Vehicle capacity (all same value) |

---

## State Initialization: DARPEnv._reset()
### File: `/Users/jiangwolin/Desktop/Research/llm-rl/rl4co/rl4co/envs/routing/darp/env.py` (Lines 252-328)

**Input:** TensorDict from generator + `batch_size=[3]`

**Processing:**

### Step 1: Extract Device and Dimensions
```python
device = td["depot"].device  # GPU or CPU
num_agents = td["capacity"].shape[-1]  # 5
num_loc = td["locs"].shape[-2]  # 20
batch_size = [3]
```

### Step 2: Concatenate Depot with Locations
```python
locs = torch.cat([td["depot"].unsqueeze(-2), td["locs"]], dim=-2)
# Before: [3, 20, 2]
# Depot:  [3, 1, 2]
# After:  [3, 21, 2]  (index 0 = depot, indices 1-20 = customers)
```

### Step 3: Initialize Vehicle State
```python
# Current node: all vehicles start at depot (node 0)
current_node = torch.zeros((3, 1), dtype=torch.int64, device=device)

# Current agent: start with agent 0
current_agent = torch.zeros((3, 1), dtype=torch.int64, device=device)

# Current time: start at time 0
current_time = torch.zeros((3, 1), dtype=torch.int64, device=device)

# Current load: vehicles start empty
current_load = torch.zeros((3, 1), dtype=torch.int64, device=device)
```

### Step 4: Initialize Global State Tracking
```python
# visited: which nodes have been visited by ANY vehicle
# Depot (index 0) not included in visited tracking (vehicles can return)
visited = torch.zeros((3, 21), dtype=torch.bool, device=device)

# picked_up_by_current: which pickups this vehicle has picked up
picked_up_by_current = torch.zeros((3, 21), dtype=torch.bool, device=device)

# vehicle_finished: which vehicles have completed their tours
vehicle_finished = torch.zeros((3, 5), dtype=torch.bool, device=device)

# Total distance accumulator
total_distance = torch.zeros((3,), dtype=torch.float32, device=device)

# Step counter
i = torch.zeros((3, 1), dtype=torch.int64, device=device)
```

### Step 5: Prepare Time Windows and Demand with Depot
```python
# Time windows: add depot (with max time 48)
time_windows = torch.cat([
    torch.full((3, 1), 48, dtype=torch.long, device=device),
    td["time_windows"]
], dim=-1)
# Shape: [3, 21] (depot at index 0 has time window 48)

# Demand: add depot (with 0 demand)
demand = torch.cat([
    torch.zeros((3, 1), dtype=torch.float32, device=device),
    td["demand"]
], dim=-1)
# Shape: [3, 21] (depot at index 0 has 0 demand)
```

### Step 6: Generate Initial Action Mask
```python
# Create temporary TensorDict with all state
td_temp = TensorDict({
    "locs": locs,                              # [3, 21, 2]
    "current_node": current_node,              # [3, 1]
    "current_agent": current_agent,            # [3, 1]
    "current_time": current_time,              # [3, 1]
    "current_load": current_load,              # [3, 1]
    "visited": visited,                        # [3, 21]
    "picked_up_by_current": picked_up_by_current,  # [3, 21]
    "vehicle_finished": vehicle_finished,      # [3, 5]
    "time_windows": time_windows,              # [3, 21]
    "demand": demand,                          # [3, 21]
    "capacity": td["capacity"],                # [3, 5]
    "i": i,                                    # [3, 1]
}, batch_size=[3])

# Call _get_action_mask to determine initial feasible actions
action_mask = self._get_action_mask(td_temp)
# Result: [3, 21] boolean tensor
```

### Step 7: Return Complete Initialization
```python
return TensorDict({
    "locs": locs,                          # [3, 21, 2]
    "current_node": current_node,          # [3, 1] - start at depot
    "current_agent": current_agent,        # [3, 1] - agent 0 is active
    "current_time": current_time,          # [3, 1] - time = 0
    "current_load": current_load,          # [3, 1] - load = 0
    "visited": visited,                    # [3, 21] - nothing visited yet
    "picked_up_by_current": picked_up_by_current,  # [3, 21] - no pickups yet
    "vehicle_finished": vehicle_finished,  # [3, 5] - no vehicles finished
    "total_distance": total_distance,      # [3] - distance = 0
    "time_windows": time_windows,          # [3, 21]
    "demand": demand,                      # [3, 21]
    "capacity": td["capacity"],            # [3, 5]
    "i": i,                                # [3, 1] - step = 0
    "action_mask": action_mask,            # [3, 21]
}, batch_size=[3])
```

---

## Initial Action Mask Generation
### File: `/Users/jiangwolin/Desktop/Research/llm-rl/rl4co/rl4co/envs/routing/darp/env.py` (Lines 330-412)

**Input State:** All zero initialization with `current_node=0` (at depot)

### Mask Computation Steps

#### Step 1: Initialize Empty Mask
```python
batch_dims = (3,)
num_loc = 20
action_mask = torch.zeros((3, 21), dtype=torch.bool, device=device)
```

#### Step 2: Compute Distances and Arrival Times
```python
# Current location: depot [3, 1, 2]
curr_loc = gather_by_index(locs, current_node)  # [3, 1, 2]

# All locations
all_locs = locs  # [3, 21, 2]

# Distance from depot to all nodes
dist = (all_locs - curr_loc).norm(p=2, dim=-1)  # [3, 21]

# Convert distance to time
travel_time = torch.round(dist / 25.0).long()  # [3, 21]

# Arrival times at each node
arrival_time = 0 + travel_time  # [3, 21]  (current_time=0)
```

#### Step 3: Define Node Types
```python
is_pickup_node = torch.tensor([
    False,  # 0: depot
    True,   # 1: pickup
    False,  # 2: dropoff
    True,   # 3: pickup
    False,  # 4: dropoff
    # ... continues for all 21 nodes
]).view(1, 21)

is_dropoff_node = torch.tensor([
    False,  # 0: depot
    False,  # 1: pickup
    True,   # 2: dropoff
    False,  # 3: pickup
    True,   # 4: dropoff
    # ... continues
]).view(1, 21)
```

#### Step 4: Compute Feasibility Terms
```python
# Time feasibility
time_feasible = arrival_time <= time_windows  # [3, 21]

# Not visited
not_visited = ~visited  # [3, 21] - all True initially

# Capacity feasibility (only for pickups)
demand = demand  # [3, 21] - positive for pickup, negative for dropoff
capacity_feasible = (0 + demand) <= capacity.unsqueeze(-1)  # [3, 21]
# For pickups: 0 + positive_demand <= 5? Usually True at start
# For dropoffs: 0 + negative_demand <= 5? Always True
```

#### Step 5: Pickup Mask
```python
pickup_mask = not_visited & time_feasible & capacity_feasible
pickup_mask = pickup_mask & is_pickup_node

# Result: [3, 21]
# True for all unvisited pickups that fit within time window and capacity
```

#### Step 6: Dropoff Mask
```python
# For dropoffs: need corresponding pickup done by CURRENT vehicle
pickup_done_by_current = torch.roll(picked_up_by_current, shifts=1, dims=-1)
pickup_done_by_current[..., 0] = False  # Depot has no corresponding pickup

# At initialization, no pickups have been done, so all False
dropoff_mask = not_visited & time_feasible & pickup_done_by_current
dropoff_mask = dropoff_mask & is_dropoff_node

# Result: [3, 21] - all False (no pickups done yet)
```

#### Step 7: Combine Pickup and Dropoff
```python
action_mask = pickup_mask | dropoff_mask  # [3, 21]
# Only pickups are feasible initially
```

#### Step 8: Depot Logic
```python
# Strategic depot return: only when load is zero and not first step
load_is_zero = (current_load == 0)  # True
at_depot = (current_node == 0)      # True
not_at_depot = ~at_depot             # False

i_val = 0  # First step
depot_feasible = (i_val > 0) & load_is_zero & not_at_depot
# (False) & (True) & (False) = False

action_mask[..., 0] = False  # Depot not accessible on first step

# Emergency exit: allow if no customer actions
no_customer_actions = ~action_mask[..., 1:].any(dim=-1)  # Will be False if pickups available
depot_emergency = no_customer_actions & (i_val > 0)     # False
action_mask[..., 0] = False | False                      # Still False
```

### Initial Action Mask Result
```python
# [3, 21] boolean mask:
# Index 0 (depot):     [False, False, False]   - Not accessible on first step
# Indices 1,3,5,...:   [True,  True,  True ]   - All feasible pickups (if capacity/time OK)
# Indices 2,4,6,...:   [False, False, False]   - No dropoffs (no pickups done yet)

# Actual values depend on:
# - Capacity constraints (whether pickup demand <= 5)
# - Time window constraints (whether arrival_time <= pickup_tw)
# - Typically most pickups are feasible at start
```

---

## Complete Output TensorDict Structure

### Final TensorDict Returned by reset()
```
TensorDict(
    batch_size=[3],
    {
        "locs": [3, 21, 2] float32
            Locations of all nodes: depot at index 0, then 20 customers
            Example: [[[47.3, 52.1], [12.4, 38.9], [28.1, 61.3], ...], ...]
        
        "current_node": [3, 1] int64
            Current node for active vehicle
            Example: [[0], [0], [0]]  (all at depot)
        
        "current_agent": [3, 1] int64
            Index of active vehicle
            Example: [[0], [0], [0]]  (agent 0 active)
        
        "current_time": [3, 1] int64
            Current time for active vehicle
            Example: [[0], [0], [0]]  (time 0)
        
        "current_load": [3, 1] int64
            Current load of active vehicle
            Example: [[0], [0], [0]]  (empty)
        
        "visited": [3, 21] bool
            Which nodes visited by ANY vehicle (excluding depot index)
            Example: [[False]*21, [False]*21, [False]*21]  (nothing visited yet)
        
        "picked_up_by_current": [3, 21] bool
            Which pickups picked up by current vehicle
            Example: [[False]*21, [False]*21, [False]*21]  (no pickups yet)
        
        "vehicle_finished": [3, 5] bool
            Which vehicles completed their tours
            Example: [[False]*5, [False]*5, [False]*5]  (no vehicles finished)
        
        "total_distance": [3] float32
            Total distance traveled so far
            Example: [0.0, 0.0, 0.0]
        
        "time_windows": [3, 21] int64
            Time deadline for each node
            - Index 0 (depot): 48
            - Pickup nodes: random in [min_pickup_tw, max_pickup_tw]
            - Dropoff nodes: based on pickup time + travel time
            Example: [[48, 15, 18, 22, 25, ...], ...]
        
        "demand": [3, 21] float32
            Demand at each node
            - Index 0 (depot): 0.0
            - Odd indices (pickups): positive [1.0, 3.0]
            - Even indices (dropoffs): negative [-1.0, -3.0]
            Example: [[0.0, 2.0, -2.0, 1.0, -1.0, ...], ...]
        
        "capacity": [3, 5] int64
            Capacity of each vehicle (all same)
            Example: [[5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]]
        
        "i": [3, 1] int64
            Step counter
            Example: [[0], [0], [0]]  (step 0)
        
        "action_mask": [3, 21] bool
            Feasible actions from current state
            - Index 0 (depot): False (not first step)
            - Pickup nodes: True if not visited, within time window, and capacity allows
            - Dropoff nodes: False (no pickups done yet)
            Example: [[False, True, False, True, False, True, ...], ...]
    }
)
```

---

## Node Index Organization

### Interleaved Structure
The generator creates nodes in interleaved pickup-dropoff pairs:

```
Index:    0    1    2    3    4    5    6    7    ...  19   20
Type:   DEPOT  P0   D0   P1   D1   P2   D2   P3   ...  P9   D9
Role:   Start  +2  -2   +1  -1   +3  -3   +2  -2   ... +1  -1
```

Where:
- P0, P1, etc. are pickups 0, 1, etc.
- D0, D1, etc. are dropoffs 0, 1, etc.
- Numbers show demand (positive for pickup, negative for dropoff)

### Demand Array Pattern
```python
demand[b] = [0, 2, -2, 1, -1, 3, -3, 2, -2, ...]
             0  1   2  3  4  5  6  7  8 
```

---

## Initial State Summary

| Component | Value | Shape | Notes |
|-----------|-------|-------|-------|
| Nodes | 21 total | - | 1 depot + 20 customers (10 pickup-dropoff pairs) |
| Vehicles | 5 | - | All start at depot, unfinished |
| Active Vehicle | 0 | - | First vehicle active |
| Location | Depot | [3, 1] | current_node = 0 |
| Time | 0 | [3, 1] | Fresh start |
| Load | 0 | [3, 1] | Empty vehicles |
| Visited Nodes | None | [3, 21] | All False initially |
| Available Actions | Pickups | [3, 21] | All feasible pickups in action_mask |
| Capacity | 5 per vehicle | [3, 5] | From generator_params |
| Total Distance | 0 | [3] | Accumulates with each step |

---

## Key Properties at Reset

### What Can Happen First?
From initial state with `action_mask[b, :] = [False, True?, False?, True?, ...]`:
- **Pickups (odd indices):** Can be visited if:
  - Not already visited
  - Arrival time <= time window
  - Current load + demand <= capacity
- **Dropoffs (even indices):** Cannot be visited (require pickup first)
- **Depot (index 0):** Cannot be visited (only on first non-zero step with zero load)

### Vehicle Switching Mechanism
Only happens when vehicle returns to depot:
1. Vehicle marks itself `vehicle_finished[b, agent] = True`
2. Next unfinished vehicle found via `_get_next_available_vehicle()`
3. New vehicle resets: time=0, load=0, picked_up_by_current=False
4. Global `visited` state is preserved across vehicles

### No State Resets
Unlike some problems, DARP doesn't reset visited state on vehicle switch:
- `visited` is global (accumulates across all vehicles)
- This enforces "no customer visited twice (by any vehicle)" constraint
- `picked_up_by_current` IS reset (vehicle-specific state)

---

## Example: Three Different Instances

When calling `env.reset(batch_size=[3])`, you get 3 completely different problem instances:

```python
Instance 0:
  - 21 nodes at specific locations (randomly sampled)
  - 10 pickup-dropoff pairs with specific demands
  - Time windows specific to this instance
  - 5 vehicles, each with capacity 5
  - All start at depot location [47.3, 52.1]

Instance 1:
  - Different 21 node locations
  - Different demands for each pair
  - Different time windows
  - Same 5 vehicles and capacity 5
  - All start at different depot location [61.8, 38.4]

Instance 2:
  - Another set of different locations
  - Another set of different demands and time windows
  - Same vehicle setup
  - All start at another depot location [29.5, 71.2]
```

All processed in parallel through batch dimension, making training efficient!

