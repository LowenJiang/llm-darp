# DARPEnv Reset Process: Quick Reference Summary

## Overview
`env.reset(batch_size=[3])` generates 3 different DARP problem instances and returns a fully initialized state ready for stepping.

---

## File Locations
- **Main Reset Logic**: `/rl4co/envs/routing/darp/env.py` lines 252-328 (`_reset` method)
- **Data Generation**: `/rl4co/envs/routing/darp/generator.py` lines 103-181 (`_generate` method)
- **Action Masking**: `/rl4co/envs/routing/darp/env.py` lines 330-412 (`_get_action_mask` method)
- **Base Class**: `/rl4co/envs/common/base.py` lines 135-143 (`RL4COEnvBase.reset` method)

---

## Call Chain (4 Steps)

### Step 1: Entry Point
```python
env.reset(batch_size=[3])
```
→ Calls `RL4COEnvBase.reset()`

### Step 2: Generate Problem Data
```python
RL4COEnvBase.reset()
  └─ td = self.generator(batch_size=[3])  # DARPGenerator._generate([3])
     Returns: TensorDict with locs, depot, demand, time_windows, capacity
```

### Step 3: Call Parent Reset
```python
RL4COEnvBase.reset()
  └─ super().reset(td, batch_size=[3])   # TorchRL EnvBase
     └─ Calls _reset(td, batch_size=[3])
```

### Step 4: Initialize State
```python
DARPEnv._reset(td, batch_size=[3])
  └─ Returns: Fully initialized TensorDict with all state fields
```

---

## Data Generation Process

### Input
- `batch_size = [3]` (3 independent problem instances)

### Sampling (DARPGenerator._generate)

| Data | Generator Parameter | Shape | Process |
|------|-------------------|-------|---------|
| **Depot** | `depot_sampler` | [3, 2] | Uniform in [min_loc, max_loc] |
| **Locations** | `loc_sampler` | [3, 20, 2] | 20 random (x,y) coordinates |
| **Demands** | `demand_sampler` | [3, 20] | Pickup +ve, dropoff -ve (interleaved) |
| **Time Windows** | Random sampling | [3, 20] | Pickup: [10, 30], Dropoff: based on travel time |
| **Capacity** | `vehicle_capacity` | [3, 5] | All 5 vehicles have capacity 5 |

### Key Patterns

**Demand Array** (interleaved structure):
```
Index:  0   1   2   3   4   5   6   7   ...
Value:  0  +2  -2  +1  -1  +3  -3  +2   ...
Type:   D   P   D   P   D   P   D   P   ...
```
- Indices 0: Depot (0 demand)
- Odd indices: Pickups (positive demand)
- Even indices: Dropoffs (negative demand = return of pickup)

**Time Windows** (pickup-dropoff constraint):
- Pickup: Random in [min_pickup_tw, max_pickup_tw]
- Dropoff: Based on pickup time + travel distance
- Ensures vehicle has time to go from pickup to dropoff

---

## State Initialization Process

### Phase 1: Merge Locations (Line 259)
```python
locs = torch.cat([td["depot"].unsqueeze(-2), td["locs"]], dim=-2)
# [3, 20, 2] + [3, 1, 2] = [3, 21, 2]
# Index 0 = depot, Indices 1-20 = customers
```

### Phase 2: Initialize Vehicle State (Lines 262-265)
All start at depot:
```python
current_node = [[0], [0], [0]]          # At depot
current_agent = [[0], [0], [0]]         # Agent 0 active
current_time = [[0], [0], [0]]          # Time 0
current_load = [[0], [0], [0]]          # Empty
```

### Phase 3: Initialize Tracking State (Lines 269-281)
```python
visited = all False [3, 21]                    # No nodes visited yet
picked_up_by_current = all False [3, 21]     # No pickups by agent
vehicle_finished = all False [3, 5]           # No vehicles finished
total_distance = [0, 0, 0]                    # No distance traveled
i = [[0], [0], [0]]                           # Step counter
```

### Phase 4: Augment with Depot Info (Lines 283-293)
Prepend depot row to demand and time_windows:
```python
time_windows = [depot_tw=48] + original[3, 20]  = [3, 21]
demand = [depot_demand=0] + original[3, 20]     = [3, 21]
```

### Phase 5: Compute Action Mask (Line 311)
```python
action_mask = _get_action_mask(td_temp)  # [3, 21] bool
# Determines feasible first actions
```

---

## Returned TensorDict Structure

### Output Shape Summary
```
TensorDict(batch_size=[3])
├─ locs: [3, 21, 2] float32           ← All node locations
├─ current_node: [3, 1] int64         ← All at depot (0)
├─ current_agent: [3, 1] int64        ← All agent 0
├─ current_time: [3, 1] int64         ← All time 0
├─ current_load: [3, 1] int64         ← All empty (0)
├─ visited: [3, 21] bool              ← All unvisited
├─ picked_up_by_current: [3, 21] bool ← No pickups yet
├─ vehicle_finished: [3, 5] bool      ← No vehicles finished
├─ total_distance: [3] float32        ← All 0.0
├─ time_windows: [3, 21] int64        ← Pickup/dropoff deadlines
├─ demand: [3, 21] float32            ← Positive (pickup), negative (dropoff)
├─ capacity: [3, 5] int64             ← All 5 per vehicle
├─ i: [3, 1] int64                    ← All step 0
└─ action_mask: [3, 21] bool          ← Feasible first actions
```

### Field Descriptions

| Field | Shape | Type | Meaning |
|-------|-------|------|---------|
| `locs` | [3,21,2] | float32 | Coordinates: idx 0=depot, 1-20=customers |
| `demand` | [3,21] | float32 | +ve=pickup, -ve=dropoff |
| `time_windows` | [3,21] | int64 | Deadline to visit each node |
| `capacity` | [3,5] | int64 | Vehicle capacity (constant) |
| `current_node` | [3,1] | int64 | Active vehicle's current node (start=0) |
| `current_agent` | [3,1] | int64 | Active vehicle index (start=0) |
| `current_time` | [3,1] | int64 | Active vehicle's current time (start=0) |
| `current_load` | [3,1] | int64 | Active vehicle's load (start=0) |
| `visited` | [3,21] | bool | Which nodes visited by ANY vehicle |
| `picked_up_by_current` | [3,21] | bool | Which pickups done by active vehicle |
| `vehicle_finished` | [3,5] | bool | Which vehicles completed tours |
| `total_distance` | [3] | float32 | Total distance so far (start=0) |
| `i` | [3,1] | int64 | Episode step counter (start=0) |
| `action_mask` | [3,21] | bool | Feasible actions from current state |

---

## Initial Action Mask Computation

### What's Feasible First?

**Feasible**: Pickups with:
- `arrival_time ≤ time_window` (time constraint)
- `current_load + demand ≤ capacity` (capacity constraint)
- Not yet visited

**Not Feasible**:
- **Depot** (index 0): Requires i > 0 (not first step)
- **Dropoffs** (even indices): Require corresponding pickup completed by current vehicle

### Mask Pattern (Initial)
```
action_mask[b] = [False, T/F, False, T/F, False, T/F, ...]
                   depot  P0   D0     P1    D1    P2  ...
```
- Depot: Always False on step 0
- Pickups: True if time/capacity/visited constraints satisfied
- Dropoffs: Always False (no pickups completed yet)

### _get_action_mask Algorithm (Lines 330-412)

```
1. Initialize mask to all False
2. Compute distances and arrival times from current location
3. Define node types (pickup vs dropoff)
4. Common feasibility:
   - time_feasible = arrival_time ≤ time_windows
   - not_visited = ~visited
   - capacity_feasible = current_load + demand ≤ capacity
5. Pickup mask: not_visited & time_feasible & capacity_feasible & is_pickup
6. Dropoff mask: not_visited & time_feasible & pickup_done_by_current & is_dropoff
7. Combine: action_mask = pickup_mask | dropoff_mask
8. Depot logic:
   - Strategic: (i > 0) & (load == 0) & (not at depot)
   - Emergency: Allow if no customer actions available
9. Return final mask
```

---

## Key Insights

### Multi-Vehicle Coordination
- **visited**: Global across all vehicles (prevents revisit by any vehicle)
- **picked_up_by_current**: Per-vehicle (resets when switching vehicles)
- **vehicle_finished**: Tracks which vehicles completed tours
- **Vehicle switching**: Only happens when vehicle returns to depot

### Time Windows
- **Pickup**: Random deadline in [10, 30]
- **Dropoff**: Constrained by pickup time + travel time
- Ensures feasibility: vehicle has time to complete pickup-dropoff

### Interleaved Structure
```
Index:  0   1   2   3   4   5
Role:  DEP  P0  D0  P1  D1  P2
Pair:   -  (0) (0) (1) (1) (2)
```
Simplifies pairing constraints and batch processing

### Batch Efficiency
All 3 instances processed in parallel through vectorized tensor operations:
- No loops over instances
- GPU-friendly operations
- Scales to large batch sizes

---

## Process Flow Diagram

```
env.reset(batch_size=[3])
    ↓
RL4COEnvBase.reset()
    ├─ Generate problem data
    │  └─ DARPGenerator._generate([3])
    │     └─ Sample locations, demands, time windows
    ├─ Call super().reset()
    │  └─ TorchRL EnvBase.reset()
    │     └─ DARPEnv._reset(td, batch_size=[3])
    │        ├─ Merge depot with locations
    │        ├─ Initialize state (all zeros/False)
    │        ├─ Augment with depot row
    │        ├─ Compute action_mask
    │        └─ Return complete TensorDict
    └─ RETURN: Initialized state ready for stepping
```

---

## First Action

From the initial state, you can step with any feasible pickup:
```python
# Example: select first feasible pickup
action = torch.argmax(action_mask[0, 1:]) + 1  # +1 to skip depot index

# Step
td = env.step(TensorDict({"action": action}, batch_size=[3]))
```

This will:
1. Move active vehicle to pickup location
2. Increase load
3. Mark pickup as completed by this vehicle
4. Mark as globally visited
5. Update time
6. Compute new action mask for next step

---

## Additional Notes

### Generator Parameters (Default)
```python
num_loc: int = 20                 # Must be even (pickup-dropoff pairs)
num_agents: int = 5               # Number of vehicles
min_loc: float = 0.0              # Coordinate range
max_loc: float = 100.0
vehicle_capacity: int = 5         # Capacity per vehicle
min_demand: int = 1               # Demand range [1, 3]
max_demand: int = 3
vehicle_speed: float = 25.0       # For time conversion
min_pickup_tw: int = 10           # Time window range
max_pickup_tw: int = 30
```

### Instance Variations
Each batch instance has:
- Different depot location
- Different node locations
- Different demands
- Different time windows
- Same vehicle count (5) and capacity (5)

All processed in parallel!

---

## Files Referenced in This Repository

1. **Main Environment**: `/rl4co/envs/routing/darp/env.py`
   - `DARPEnv` class
   - `_reset()` method (lines 252-328)
   - `_get_action_mask()` method (lines 330-412)

2. **Data Generation**: `/rl4co/envs/routing/darp/generator.py`
   - `DARPGenerator` class
   - `_generate()` method (lines 103-181)

3. **Base Class**: `/rl4co/envs/common/base.py`
   - `RL4COEnvBase` class
   - `reset()` method (lines 135-143)

4. **Utilities**: `/rl4co/envs/common/utils.py`
   - `Generator` base class
   - `get_sampler()` function

5. **Documentation**: 
   - `DARP_RESET_ANALYSIS.md` (this directory)
   - `DARP_RESET_FLOW_DIAGRAM.md` (this directory)
   - `STEP_AND_MASKING_LOGIC.md` (this directory)

---

## Summary

The reset process is a well-orchestrated initialization sequence:
1. **Generation**: Random sampling of problem data (locations, demands, time windows)
2. **Merging**: Combining depot with customer locations
3. **State Init**: Zero-initialization of all state variables
4. **Masking**: Computing feasible actions based on constraints
5. **Return**: Complete TensorDict ready for stepping

The result is a fully initialized batched environment state where:
- All vehicles start at the depot with empty load
- No customers have been visited
- Time is 0
- Feasible first actions are all pickups within time window and capacity
- All 3 batch instances are ready for stepping simultaneously
