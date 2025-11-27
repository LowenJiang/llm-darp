# DARPEnv Reset Process: Visual Flow Diagram

## High-Level Call Flow

```
env.reset(batch_size=[3])
    ↓
RL4COEnvBase.reset()
    ├─ Normalizes batch_size → [3]
    ├─ td = self.generator(batch_size=[3])
    │   ↓
    │   Generator.__call__()
    │   ├─ Calls DARPGenerator._generate([3])
    │   └─ Returns TensorDict with:
    │       ├─ locs: [3, 20, 2]
    │       ├─ depot: [3, 2]
    │       ├─ demand: [3, 20]
    │       ├─ time_windows: [3, 20]
    │       └─ capacity: [3, 5]
    │
    ├─ super().reset(td, batch_size=[3])
    │   ↓
    │   TorchRL EnvBase.reset()
    │   └─ Calls _reset(td, batch_size=[3])
    │       ↓
    │       DARPEnv._reset()
    │       └─ Returns complete initialized state
    │
    └─ RETURNS: TensorDict with all state fields
```

---

## Data Generation Flow

```
DARPGenerator._generate([3])
├─ Input: batch_size = [3]
│
├─ 1. Sample depot locations
│  ├─ depot_sampler.sample([3, 2])
│  └─ Result: [3, 2]
│      [[47.3, 52.1],
│       [61.8, 38.4],
│       [29.5, 71.2]]
│
├─ 2. Sample node locations
│  ├─ loc_sampler.sample([3, 20, 2])
│  └─ Result: [3, 20, 2]
│      Random coordinates for each node
│
├─ 3. Sample and create demands
│  ├─ demand_sampler.sample([3, 10])     # One per pickup
│  ├─ Convert to [min_demand, max_demand] range
│  └─ Create full demand tensor with negatives for dropoffs
│      Result: [3, 20] with pattern [+2, -2, +1, -1, +3, -3, ...]
│
├─ 4. Generate time windows
│  ├─ For each pickup-dropoff pair:
│  │  ├─ pickup_tw = random(min_pickup_tw, max_pickup_tw)
│  │  ├─ distance = |pickup_location - dropoff_location|
│  │  ├─ travel_time = distance / vehicle_speed
│  │  └─ dropoff_tw = random(pickup_tw + travel_time,
│  │                          pickup_tw + 2*travel_time)
│  └─ Result: [3, 20]
│
├─ 5. Create capacity tensor
│  ├─ capacity = [vehicle_capacity] × num_agents
│  └─ Result: [3, 5] with all values = 5
│
└─ Return TensorDict(locs, depot, demand, time_windows, capacity)
```

---

## State Initialization Flow

```
DARPEnv._reset(td, batch_size=[3])
├─ Input: td from generator
│  └─ Contains: locs[3,20,2], depot[3,2], demand[3,20], 
│               time_windows[3,20], capacity[3,5]
│
├─ 1. Merge depot with locations
│  ├─ Concatenate: [depot] + [locs]
│  └─ Result: locs[3,21,2] (index 0 = depot)
│
├─ 2. Initialize vehicle state (all vehicles)
│  ├─ current_node = [0, 0, 0]           → Start at depot
│  ├─ current_agent = [0, 0, 0]          → Agent 0 active
│  ├─ current_time = [0, 0, 0]           → Time 0
│  └─ current_load = [0, 0, 0]           → Empty
│
├─ 3. Initialize tracking state
│  ├─ visited[3,21] = all False          → No nodes visited
│  ├─ picked_up_by_current[3,21] = all False → No pickups
│  ├─ vehicle_finished[3,5] = all False   → No finished
│  ├─ total_distance[3] = [0, 0, 0]      → No distance
│  └─ i[3,1] = [0, 0, 0]                 → Step counter
│
├─ 4. Prepare augmented tensors
│  ├─ Prepend depot to time_windows
│  │  └─ Result: [3,21] with depot TW = 48
│  └─ Prepend depot (0 demand) to demand
│      └─ Result: [3,21] with depot demand = 0
│
├─ 5. Compute initial action mask
│  ├─ Create temp TensorDict with all state
│  ├─ Call _get_action_mask(td_temp)
│  │  ├─ Compute distances from depot to all nodes
│  │  ├─ Check time window feasibility
│  │  ├─ Check capacity feasibility
│  │  ├─ Check pickup requirements
│  │  └─ Result: [3,21] bool mask
│  └─ Initial mask: only pickups feasible (no depot on first step)
│
└─ Return complete TensorDict with all state fields
```

---

## Action Mask Computation

```
_get_action_mask(state)
├─ Initialize: mask[3,21] = all False
│
├─ Compute distances & arrival times
│  ├─ Current location: depot [3, 1, 2]
│  ├─ Distance to each node: [3, 21]
│  ├─ Travel time: distance / vehicle_speed: [3, 21]
│  └─ Arrival time: current_time + travel_time: [3, 21]
│
├─ Define node types
│  ├─ is_pickup = [F, T, F, T, F, T, ...] (odd indices)
│  └─ is_dropoff = [F, F, T, F, T, F, ...] (even, non-zero)
│
├─ Compute common feasibility
│  ├─ time_feasible = (arrival_time ≤ time_windows): [3, 21]
│  ├─ not_visited = (~visited): [3, 21]
│  └─ capacity_feasible = (load + demand ≤ capacity): [3, 21]
│
├─ Pickup mask
│  ├─ pickup_mask = not_visited & time_feasible & capacity_feasible
│  ├─ pickup_mask = pickup_mask & is_pickup_node
│  └─ Result: [3, 21] - True for feasible pickups
│
├─ Dropoff mask
│  ├─ For each dropoff, check if pickup done by CURRENT vehicle
│  ├─ pickup_done_by_current = roll(picked_up_by_current, 1)
│  ├─ dropoff_mask = ... & pickup_done_by_current & is_dropoff_node
│  └─ Result: [3, 21] - True for feasible dropoffs
│
├─ Combine pickup & dropoff
│  └─ action_mask = pickup_mask | dropoff_mask
│
├─ Depot logic
│  ├─ Regular depot:
│  │  └─ depot_feasible = (step > 0) & (load == 0) & (not at depot)
│  ├─ Emergency depot:
│  │  ├─ no_customer_actions = ~any(action_mask[1:])
│  │  └─ Allow depot if no other options and step > 0
│  └─ action_mask[0] = regular depot | emergency depot
│
└─ Return: action_mask[3,21]
```

---

## TensorDict Structure at Each Stage

### Stage 1: After Generator
```
TensorDict (batch_size=[3])
├─ locs: [3, 20, 2]          float32
├─ depot: [3, 2]              float32
├─ demand: [3, 20]            float32
├─ time_windows: [3, 20]      int64
└─ capacity: [3, 5]           int64
```

### Stage 2: Before Action Mask (in _reset)
```
TensorDict (batch_size=[3])
├─ locs: [3, 21, 2]                      ← Merged with depot
├─ current_node: [3, 1]                  ← All 0
├─ current_agent: [3, 1]                 ← All 0
├─ current_time: [3, 1]                  ← All 0
├─ current_load: [3, 1]                  ← All 0
├─ visited: [3, 21]                      ← All False
├─ picked_up_by_current: [3, 21]        ← All False
├─ vehicle_finished: [3, 5]              ← All False
├─ time_windows: [3, 21]                 ← Prepended depot
├─ demand: [3, 21]                       ← Prepended depot
└─ capacity: [3, 5]
```

### Stage 3: Final Reset Output
```
TensorDict (batch_size=[3])
├─ locs: [3, 21, 2]                      float32
├─ current_node: [3, 1]                  int64
├─ current_agent: [3, 1]                 int64
├─ current_time: [3, 1]                  int64
├─ current_load: [3, 1]                  int64
├─ visited: [3, 21]                      bool
├─ picked_up_by_current: [3, 21]        bool
├─ vehicle_finished: [3, 5]              bool
├─ total_distance: [3]                   float32
├─ time_windows: [3, 21]                 int64
├─ demand: [3, 21]                       float32
├─ capacity: [3, 5]                      int64
├─ i: [3, 1]                             int64
└─ action_mask: [3, 21]                  bool    ← Last computed
```

---

## Example Batch Element (Instance 0)

### Locations (locs[0])
```
Index:  0         1           2           3           ...  20
Type:   DEPOT     PICKUP_0    DROPOFF_0   PICKUP_1        DROPOFF_9
Coord: [47.3,    [12.4,      [28.1,      [35.6,          [62.7,
        52.1]     38.9]       61.3]       41.2]           19.8]
```

### Demands (demand[0])
```
Index:  0    1    2    3    4     5     6     7    ...  19    20
Value:  0.0  +2   -2   +1   -1    +3    -3    +2       +1    -1
Type:   D    P    D    P    D     P     D     P        P     D
```

### Time Windows (time_windows[0])
```
Index:  0    1    2    3    4     5     6     7    ...  19    20
Value:  48   15   18   22   25    12    17    20       14    21
```

### Initial State (step 0)
```
current_node: 0         (At depot)
current_agent: 0        (Agent 0 active)
current_time: 0         (Time = 0)
current_load: 0         (Load = 0)
visited: [F,F,...,F]    (Nothing visited)
picked_up_by_current: [F,F,...,F]  (No pickups)
vehicle_finished: [F,F,F,F,F]      (No finished)
total_distance: 0.0
i: 0                    (Step 0)
```

### Initial Action Mask (step 0)
```
Index:  0     1     2     3     4     5     ...  19    20
Can:    False True  False True  False True        True  False
Type:   D     P     D     P     D     P          P     D
Reason: -     ✓     -     ✓     -     ✓          ✓     -

All feasible pickups available (assuming time window & capacity OK)
Depot NOT accessible (i > 0 required)
No dropoffs (no pickups completed by current vehicle yet)
```

---

## Time Window Example

For a pickup-dropoff pair:
```
Pickup location: [12.4, 38.9]
Dropoff location: [28.1, 61.3]
Distance: 26.3 units
Travel time: 26.3 / 25.0 = 1.052 ≈ 1 time unit

Sampled pickup time window: 15
Sampled dropoff time window: random(15+1, 15+2) = random(16, 17)

So dropoff time window ≈ 16 or 17

This ensures:
- Vehicle can pick up by time 15
- Vehicle can travel (1 time unit) and drop off by time 16-17
```

---

## Vehicle Switching Scenario (Not in reset, but shows state preservation)

```
Before Depot Return:
├─ current_agent: 0
├─ current_node: Some customer node
├─ visited: [T,T,T,F,T,F,...] (global - includes all vehicles)
├─ picked_up_by_current: [F,T,F,T,F,...] (agent 0's pickups)
└─ vehicle_finished: [F,F,F,F,F]

After Visit Depot (depot return):
├─ Agent 0 marked finished: vehicle_finished: [T,F,F,F,F]
├─ Current node reset to depot: current_node: 0
├─ Current agent switched to 1: current_agent: 1
├─ Agent 1 state reset:
│  ├─ current_time: 0 (fresh start)
│  ├─ current_load: 0 (empty)
│  └─ picked_up_by_current: [F,F,...,F] (no pickups yet)
│
└─ GLOBAL STATE PRESERVED:
   └─ visited: [T,T,T,F,T,F,...] (unchanged - enforces no revisit)
```

---

## Batch Processing Parallelism

```
Reset for batch_size=[3]:

Instance 0:          Instance 1:           Instance 2:
Depot: [47.3, ...]  Depot: [61.8, ...]   Depot: [29.5, ...]
Locs: 20 nodes      Locs: 20 nodes       Locs: 20 nodes
Demands: [+2,-2,]   Demands: [+3,-3,]    Demands: [+1,-1,]
...                 ...                  ...

     ↓                    ↓                    ↓
   All processed in parallel through batched tensor operations
     ↓

Result: All 3 instances initialized simultaneously
├─ locs[3, 21, 2]
├─ action_mask[3, 21]
└─ All other state[3, ...]
```

