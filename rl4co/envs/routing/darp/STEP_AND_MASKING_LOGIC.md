# DARP Environment: Step and Action Masking Logic

## Overview

This document describes the step transition logic (`_step`) and action masking logic (`_get_action_mask`) for the Dial-a-Ride Problem (DARP) environment in RL4CO.

## Problem Structure

### Node Layout (Interleaved)
- **Index 0**: Depot
- **Odd indices (1, 3, 5, ...)**: Pickup locations
- **Even indices (2, 4, 6, ...)**: Dropoff locations
- **Pairing**: Pickup at index `i` corresponds to dropoff at index `i+1`

### Multi-Vehicle Routing
- Multiple vehicles (`num_agents`) serve requests
- Each vehicle makes **ONE tour** (can only return to depot once)
- When a vehicle returns to depot, it's marked as finished and the next available vehicle takes over
- All vehicles start and end at the depot

## `_step()` Method Logic

### Input
- `td["action"]`: The selected node index to visit next

### Step-by-Step Process

#### 1. Update Visited Status (Lines 84-85)
```python
visited = td["visited"].scatter(-1, current_node, 1)
```
- Marks the current node as globally visited (by ANY vehicle)
- Depot is NOT marked as visited (vehicles can return multiple times with different vehicles)

#### 2. Calculate Travel Time (Lines 87-94)
```python
prev_loc = gather_by_index(td["locs"], td["current_node"])
curr_loc = gather_by_index(td["locs"], current_node)
dist = torch.norm(curr_loc - prev_loc, p=2, dim=-1, keepdim=True)
travel_time = torch.round(dist / self.generator.vehicle_speed).long()
current_time = td["current_time"] + travel_time
```
- Computes Euclidean distance between previous and current location
- Converts to travel time using vehicle speed
- Updates current time

#### 3. Update Load and Pickup Status (Lines 96-113)
**For Pickup Nodes** (odd indices):
```python
is_pickup = (current_node % 2 == 1)
if is_pickup.any():
    demand = gather_by_index(td["demand"], current_node)
    current_load = current_load + demand  # Increase load
    picked_up_by_current = picked_up_by_current.scatter(-1, current_node, 1)
```
- Increases vehicle load by pickup demand
- Marks pickup as completed by **current vehicle** (important for pairing constraint)

**For Dropoff Nodes** (even indices, excluding depot):
```python
is_dropoff = (current_node % 2 == 0) & (current_node > 0)
if is_dropoff.any():
    demand = gather_by_index(td["demand"], current_node)
    current_load = current_load + demand  # Demand is negative, so this decreases load
```
- Decreases vehicle load (demand is negative for dropoffs)

#### 4. Track Total Distance (Line 116)
```python
total_distance = td["total_distance"] + dist.squeeze(-1)
```
- Accumulates total distance for reward calculation

#### 5. Handle Vehicle Switching (Lines 118-176)

**Detect Depot Return** (Lines 119-122):
```python
prev_at_depot = td["current_node"] == 0
now_at_depot = current_node == 0
returning_to_depot = now_at_depot & ~prev_at_depot
```
- Identifies when vehicle returns to depot from a non-depot location

**Mark Vehicle as Finished** (Lines 136-140):
```python
if returning_to_depot.any():
    batch_indices = torch.arange(vehicle_finished.shape[0], device=device)
    agent_indices = current_agent.squeeze(-1).long()
    vehicle_finished[batch_indices, agent_indices] = True
```
- When vehicle returns to depot, it's permanently marked as finished

**Switch to Next Vehicle** (Lines 142-176):
- Checks if there are unvisited customers
- Finds next available (unfinished) vehicle using `_get_next_available_vehicle()`
- If switching occurs:
  - Resets `current_agent` to next vehicle
  - Resets `current_time` to 0
  - Resets `current_load` to 0
  - Clears `picked_up_by_current` (new vehicle has no pickups yet)
  - Resets `current_node` to depot (0)

#### 6. Generate Action Mask (Lines 178-189)
```python
action_mask = self._get_action_mask(td.update({...}))
```
- Determines which actions are feasible for the new state

#### 7. Check Termination (Lines 191-198)
```python
all_visited = visited[..., 1:].all(dim=-1)
all_vehicles_finished = vehicle_finished.all(dim=-1)
done = all_visited | all_vehicles_finished
no_actions_available = ~action_mask.any(dim=-1)
done = done | no_actions_available
```
Episode ends when:
- All customers are visited, OR
- All vehicles have finished their tours, OR
- No actions are available (deadlock/infeasible state)

#### 8. Update TensorDict (Lines 203-217)
Returns updated state with all new values.

---

## `_get_action_mask()` Method Logic

### Purpose
Generate a boolean mask indicating which nodes are feasible to visit next, considering:
1. Time window constraints
2. Capacity constraints
3. Pickup-before-dropoff constraints
4. Same-vehicle pairing constraints
5. Strategic depot return rules

### Step-by-Step Process

#### 1. Initialize (Lines 332-347)
```python
action_mask = torch.zeros((*batch_dims, num_loc + 1), dtype=torch.bool, device=device)
```
- Start with all actions masked (False)
- Extract and normalize state variables

#### 2. Compute Distances and Arrival Times (Lines 352-360)
```python
curr_loc = td["locs"].gather(-2, current_node.unsqueeze(-1).expand(...))
all_locs = td["locs"]
dist = (all_locs - curr_loc).norm(p=2, dim=-1)
travel_time = torch.round(dist / self.generator.vehicle_speed).long()
arrival_time = current_time.unsqueeze(-1) + travel_time
```
- Vectorized computation for ALL nodes at once
- Computes when vehicle would arrive at each node

#### 3. Define Node Types (Lines 362-367)
```python
is_pickup_node = (idx % 2 == 1)
is_dropoff_node = ((idx % 2 == 0) & (idx != 0))
```
- Creates boolean masks for node types

#### 4. Common Feasibility Terms (Lines 369-375)
```python
time_feasible = arrival_time <= td["time_windows"]
not_visited = ~visited
capacity_feasible = (current_load + demand) <= capacity
```
- Time: Can arrive before deadline
- Not visited: Node hasn't been visited by any vehicle
- Capacity: Vehicle has room for pickup demand

#### 5. Pickup Mask (Lines 377-379)
```python
pickup_mask = not_visited & time_feasible & capacity_feasible
pickup_mask = pickup_mask & is_pickup_node
```
**Pickup is feasible if**:
- Node is a pickup node (odd index)
- Not yet visited
- Can arrive within time window
- Vehicle has sufficient capacity

#### 6. Dropoff Mask (Lines 381-386)
```python
pickup_done_by_current = torch.roll(picked_up_by_current, shifts=1, dims=-1)
pickup_done_by_current[..., 0] = False
dropoff_mask = not_visited & time_feasible & pickup_done_by_current
dropoff_mask = dropoff_mask & is_dropoff_node
```
**Dropoff is feasible if**:
- Node is a dropoff node (even index, not depot)
- Not yet visited
- Can arrive within time window
- **Corresponding pickup was done by CURRENT vehicle** (pairing constraint)
  - For dropoff at index `i`, corresponding pickup is at index `i-1`
  - Uses `torch.roll` to shift `picked_up_by_current` by 1

#### 7. Combine Pickup and Dropoff (Line 389)
```python
action_mask = action_mask | pickup_mask | dropoff_mask
```
- Union of feasible pickups and dropoffs

#### 8. Depot Logic (Lines 391-410)

**Strategic Depot Return** (Lines 392-401):
```python
load_is_zero = (current_load == 0)
at_depot = (current_node.squeeze(-1) == 0)
not_at_depot = ~at_depot
depot_feasible = (i_val > 0) & load_is_zero & not_at_depot
action_mask[..., 0] = depot_feasible
```
**Depot is feasible if**:
- Not the first step (`i > 0`)
- Load is zero (all picked-up deliveries completed)
- NOT currently at depot (prevents self-loops; returning to depot ends tour)

**Emergency Depot Exit** (Lines 403-410):
```python
no_customer_actions = ~action_mask[..., 1:].any(dim=-1)
depot_emergency = no_customer_actions & (i_val > 0)
action_mask[..., 0] = action_mask[..., 0] | depot_emergency
```
**Allow depot as last resort if**:
- No customer actions are available
- Not the first step
- Prevents deadlock in infeasible instances
- Even allows depot self-loop (will trigger `done` in `_step`)

### Key Constraints Summary

| Constraint | Implementation |
|------------|----------------|
| **Time Windows** | `arrival_time <= td["time_windows"]` |
| **Capacity** | `current_load + demand <= capacity` (pickups only) |
| **No Revisit** | `~visited` (global across all vehicles) |
| **Pickup Before Dropoff** | Dropoff requires corresponding pickup in `picked_up_by_current` |
| **Same Vehicle Pairing** | Only current vehicle's pickups enable dropoffs |
| **Zero Load at Depot** | `current_load == 0` for strategic return |
| **No Depot Self-Loop** | `not_at_depot` (except emergency) |
| **One Tour Per Vehicle** | Depot return marks vehicle as finished |

---

## Multi-Vehicle Coordination

### Vehicle State Tracking
- `current_agent`: Index of active vehicle [0, num_agents-1]
- `vehicle_finished`: Boolean tensor tracking which vehicles completed tours
- `picked_up_by_current`: Resets when switching vehicles (vehicle-specific state)
- `visited`: Global state across all vehicles (never resets)

### Vehicle Switching Mechanism
```python
def _get_next_available_vehicle(vehicle_finished, num_agents, current_agent):
    # Find first unfinished vehicle
    for v in range(num_agents):
        if not vehicle_finished[b, v]:
            return v
    return -1  # No vehicles available
```
- Sequential search for next unfinished vehicle
- Returns -1 if all vehicles are finished

### State Reset on Switch
When switching to a new vehicle after depot return:
1. `current_agent` → next available vehicle index
2. `current_time` → 0 (new tour starts fresh)
3. `current_load` → 0 (empty vehicle)
4. `picked_up_by_current` → all False (no pickups yet)
5. `current_node` → 0 (start at depot)

**Global state preserved**:
- `visited`: Maintains which customers were served by ANY vehicle
- `total_distance`: Continues accumulating across all vehicles

---

## Reward Calculation

### Implementation (Lines 418-426)
```python
def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
    num_unvisited = (~td["visited"][..., 1:]).sum(dim=-1).float()
    cost = td["total_distance"] + self.penalty_unvisited * num_unvisited
    return -cost
```

### Components
1. **Total Distance**: Sum of all travel distances across all vehicles
2. **Penalty Term**: `penalty_unvisited` (default: 100.0) × number of unvisited customers
3. **Reward**: Negative of total cost (minimization problem)

### Notes
- Reward is 0 during episode (Line 201), only computed at end via `get_reward`
- Allows partial solutions (not all customers served) with penalty
- Handles infeasible instances gracefully

---

## Edge Cases and Safety Mechanisms

### 1. Infeasible Instances
- If no customer actions available, allow emergency depot return
- Prevents infinite loops and training crashes
- Penalized via `penalty_unvisited` in reward

### 2. Deadlock Prevention
- Emergency depot action ensures at least one action always available (after step 0)
- Episode marked as done if no actions available

### 3. Vehicle Exhaustion
- Episode ends when all vehicles finished, even if customers remain
- Remaining customers penalized in reward

### 4. Depot Self-Loop Prevention
- Strategic depot return requires `not_at_depot`
- Prevents vehicle from staying at depot indefinitely
- Only allowed in emergency (no other actions)

### 5. Load Consistency
- Depot return only when `load == 0` (strategic)
- Emergency return may violate this (logged as warning in validation)

---

## Vectorization and Efficiency

### Batched Operations
- All computations support arbitrary batch dimensions: `[B]` or `[B, S]`
- Vectorized distance/time calculations for all nodes simultaneously (Line 358)
- No loops over nodes in masking logic

### Shape Conventions
- Scalar state (time, load, agent): `[..., 1]` or `[...]`
- Node-indexed state (visited, demand): `[..., N]` where N = num_loc + 1
- Multi-agent state (capacity, finished): `[..., num_agents]`

### Performance Notes
- `_get_action_mask` is fully vectorized except node type helpers
- `_get_next_available_vehicle` uses Python loops (could be optimized)
- TensorDict scatter/gather operations are efficient for GPU

---

## Differences from Single-Vehicle VRP

| Aspect | Single-Vehicle VRP | Multi-Vehicle DARP |
|--------|-------------------|-------------------|
| Depot visits | Multiple allowed | One per vehicle (ends tour) |
| State reset | Never | On vehicle switch |
| Pickup tracking | Global | Per-vehicle (`picked_up_by_current`) |
| Termination | All visited | All visited OR all vehicles done |
| Capacity | Reset at depot | Reset on vehicle switch |
| Time | Reset at depot | Reset on vehicle switch |

---

## Summary

The DARP environment implements a sophisticated multi-vehicle routing system with:

1. **Strict pairing constraints**: Pickup-dropoff pairs must be served by the same vehicle in order
2. **Multi-vehicle coordination**: Sequential vehicle deployment with state isolation
3. **Time-sensitive routing**: Time windows enforced via arrival time calculations
4. **Capacity management**: Vehicle load tracked per-vehicle with zero-load depot returns
5. **Robust masking**: Handles feasible and infeasible instances gracefully
6. **Efficient computation**: Fully vectorized operations for scalability

The key innovation is the **per-vehicle pickup tracking** (`picked_up_by_current`) combined with **global visited status**, enabling correct pairing constraints across multiple vehicles while preventing duplicate visits.
