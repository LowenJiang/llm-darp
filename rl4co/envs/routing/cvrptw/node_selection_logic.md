# CVRPTW Node Selection Logic

This document explains the node selection mechanism in the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) environment.

## Overview

The CVRPTW environment extends CVRP by adding time window constraints. At each step, the agent selects a node (customer or depot) to visit next, subject to capacity, visit, and time window constraints.

## Key Components

### 1. Action Masking (`get_action_mask()`)

The action mask determines which nodes are valid for selection at the current state. Located in `env.py:100-111`.

```python
def get_action_mask(td: TensorDict) -> torch.Tensor:
    # Inherit CVRP constraints (capacity + visited)
    not_masked = CVRPEnv.get_action_mask(td)

    # Calculate distances from current location to all nodes
    current_loc = gather_by_index(td["locs"], td["current_node"])
    dist = get_distance(current_loc[..., None, :], td["locs"])

    # Time window constraint: can only visit if arrival time <= end of time window
    can_reach_in_time = (
        td["current_time"] + dist <= td["time_windows"][..., 1]
    )

    return not_masked & can_reach_in_time
```

**Constraints Applied:**

1. **CVRP Constraints** (inherited from parent class):
   - **Capacity**: `demand + used_capacity <= vehicle_capacity` (CVRP env.py:135)
   - **Visited**: Node must not have been visited already (CVRP env.py:138)
   - **Depot**: Cannot return to depot immediately after visiting it if unserved nodes remain (CVRP env.py:141-143)

2. **Time Window Constraint** (CVRPTW-specific):
   - **Arrival deadline**: `current_time + distance <= time_window_end`
   - The vehicle must **start** service before the time window closes (not necessarily finish)

### 2. State Update (`_step()`)

After an action is selected, the state is updated. Located in `env.py:113-130`.

```python
def _step(self, td: TensorDict) -> TensorDict:
    batch_size = td["locs"].shape[0]

    # Get travel distance to selected node
    distance = gather_by_index(td["distances"], td["action"])

    # Get service duration at selected node
    duration = gather_by_index(td["durations"], td["action"])

    # Get time window start time
    start_times = gather_by_index(td["time_windows"], td["action"])[..., 0]

    # Update current time
    td["current_time"] = (td["action"][:, None] != 0) * (
        torch.max(td["current_time"] + distance, start_times) + duration
    )

    # Update other state variables (capacity, visited, etc.) via parent class
    td = super()._step(td)

    return td
```

**Time Update Logic:**

- **If depot selected** (`action == 0`): Current time resets to 0
- **If customer selected** (`action != 0`):
  1. Calculate arrival time: `current_time + travel_distance`
  2. Account for waiting: `max(arrival_time, time_window_start)`
  3. Add service duration: `service_start_time + duration`
  4. This becomes the new `current_time`

The formula ensures:
- Vehicle waits if it arrives before the time window opens
- Service completes before leaving for the next node

### 3. State Initialization (`_reset()`)

The initial state is set up in `env.py:132-161`:

```python
def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
    td_reset = TensorDict({
        "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),  # Depot at index 0
        "demand": td["demand"],
        "current_node": torch.zeros(...),  # Start at depot
        "current_time": torch.zeros(...),  # Start at time 0
        "used_capacity": torch.zeros(...),
        "vehicle_capacity": torch.full(..., self.generator.vehicle_capacity),
        "visited": torch.zeros(...),  # No nodes visited yet
        "durations": td["durations"],
        "time_windows": td["time_windows"],
    }, batch_size=batch_size)

    td_reset.set("action_mask", self.get_action_mask(td_reset))
    return td_reset
```

**Initial State:**
- Vehicle starts at depot (node 0) at time 0
- Zero capacity used, no nodes visited
- Action mask computed to show initially reachable nodes

## Node Selection Process

The complete node selection process follows these steps:

1. **State Observation**: Agent observes current state (location, time, capacity, visited nodes)

2. **Action Mask Computation**: Environment computes which nodes are feasible:
   - Not already visited
   - Sufficient capacity remaining OR is depot
   - Can reach before time window closes
   - Not depot if just visited depot and customers remain

3. **Policy Decision**: RL policy (e.g., Attention Model) selects from masked valid actions

4. **State Transition**: Environment updates state based on selected action:
   - Update current location
   - Update time (travel + wait + service)
   - Update capacity
   - Mark node as visited
   - Recompute action mask

5. **Termination Check**: Episode ends when all customers visited and vehicle returns to depot

## Constraint Hierarchy

The constraints are enforced in this order:

```
CVRPTW Action Mask = CVRP Constraints AND Time Window Constraints

CVRP Constraints:
├─ Not previously visited
├─ Capacity sufficient (or is depot)
└─ Not depot if just left depot with customers remaining

Time Window Constraints:
└─ current_time + distance_to_node <= node.time_window_end
```

## Example Scenario

Consider state:
- Current location: Customer 3
- Current time: 25
- Used capacity: 40/100
- Unvisited customers: {1, 2, 5}

For Customer 1:
- Distance from Customer 3: 5 units
- Time window: [10, 35]
- Demand: 20

**Feasibility Check:**
1. Not visited? ✓ (Customer 1 unvisited)
2. Capacity OK? ✓ (40 + 20 = 60 ≤ 100)
3. Time feasible? ✓ (25 + 5 = 30 ≤ 35)

**Result:** Customer 1 is a valid action

**If selected:**
- New location: Customer 1
- New time: max(25 + 5, 10) + duration = 30 + duration
- New capacity: 60

## Time-Infeasible Nodes

**Q: What happens to nodes that become time-infeasible? Does that mean only a subset of nodes are visited?**

**A: No, all customers must still   visited.** The environment has two key mechanisms to ensure this:

### 1. Instance Generation Guarantees Feasibility

The `CVRPTWGenerator` (in `generator.py:77-157`) creates instances that are **always feasible**:

```python
# Time windows are bounded to ensure feasibility
upper_bound = self.max_time - dist - durations  # Line 94

# Time window ends are set to ensure vehicle can return to depot
min_ts = (dist + (upper_bound - dist) * ts_1).int()
max_ts = (dist + (upper_bound - dist) * ts_2).int()
```

This ensures every customer has a time window that:
- Starts no earlier than the travel time from depot
- Ends early enough that the vehicle can return to depot before `max_time`

### 2. Depot Resets Time to Zero

When a node becomes temporarily unreachable due to time constraints, the vehicle can:
1. **Return to depot** (which resets `current_time = 0`)
2. **Start a new route** with fresh time budget
3. **Visit previously time-infeasible nodes**

This is why depot visits reset time (line 125 in `env.py:_step()`):
```python
td["current_time"] = (td["action"][:, None] != 0) * (...)
# If action == 0 (depot), current_time becomes 0
```

### Example: Handling Time-Infeasible Nodes

```
Current state:
- Current time: 200
- At customer 5
- Unvisited: {Customer 7, Customer 8}

Customer 7:
- Distance from Customer 5: 30
- Time window: [50, 90]
- Check: 200 + 30 = 230 > 90 ❌ INFEASIBLE

Customer 8:
- Distance from Customer 5: 20
- Time window: [210, 250]
- Check: 200 + 20 = 220 ≤ 250 ✓ FEASIBLE

Action sequence:
1. Visit Customer 8 (still reachable at time 220)
2. Return to depot (time resets to 0)
3. Visit Customer 7 (now feasible: 0 + distance ≤ 90)
```

### 3. Validation Enforces Complete Visitation

The `check_solution_validity()` function verifies all customers are visited:

```python
# From CVRP parent class (cvrp/env.py:158-170)
assert (torch.arange(1, graph_size + 1, ...) == sorted_pi[:, -graph_size:]).all()
```

This ensures the solution includes all customer nodes exactly once.

## Key Design Decisions

1. **Time Window End Constraint Only**: The mask only checks if service can **start** before deadline, not finish. Service duration is added after arrival.

2. **Waiting is Implicit**: If vehicle arrives early, it waits until time window opens (handled in `_step()`)

3. **Depot Time Reset**: Returning to depot resets time to 0 (line 125 in `_step()` and line 214 in `check_solution_validity()`). This allows the vehicle to visit nodes that became time-infeasible on previous routes.

4. **Distance Pre-computation**: Distances are computed once in `get_action_mask()` and stored in `td["distances"]` for reuse in `_step()`

5. **Feasible Instance Generation**: The generator ensures all instances are feasible by construction, so every customer can be reached within time windows across multiple depot visits.

## References

- Parent class CVRP logic: `/rl4co/envs/routing/cvrp/env.py:133-144`
- CVRPTW environment: `/rl4co/envs/routing/cvrptw/env.py`
- Validation logic: `check_solution_validity()` in `env.py:169-214`
