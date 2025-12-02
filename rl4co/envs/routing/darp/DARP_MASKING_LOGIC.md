# DARP Node Masking Logic

## Overview

The Dial-a-Ride Problem (DARP) environment implements sophisticated action masking in `_get_action_mask()` (lines 253-334) to ensure feasible solutions that respect pickup-dropoff pairing, time windows, and vehicle capacity constraints.

## Node Structure

- **Node 0**: Depot
- **Odd indices (1, 3, 5, ...)**: Pickup locations
- **Even indices (2, 4, 6, ...)**: Dropoff locations (interleaved pattern)

## Masking Rules

### 1. Pickup Node Masking (lines 300-302)

A pickup node can be visited if ALL conditions are met:

```python
pickup_mask = not_visited & time_feasible & capacity_feasible & is_pickup_node
```

- **Not visited**: Node hasn't been visited yet
- **Time feasible**: `arrival_time <= time_window` deadline
- **Capacity feasible**: `current_load + demand <= vehicle_capacity`
- **Is pickup node**: Node index is odd (interleaved structure)

### 2. Dropoff Node Masking (lines 304-308)

A dropoff node can be visited if ALL conditions are met:

```python
dropoff_mask = not_visited & time_feasible & pickup_done & is_dropoff_node
```

- **Not visited**: Node hasn't been visited yet
- **Time feasible**: Arrival time within time window
- **Pickup done**: Corresponding pickup (previous odd index) has been picked up by the current vehicle
- **Is dropoff node**: Node index is even and not depot

**Key constraint**: Uses `torch.roll(picked_up, shifts=1, dims=-1)` to check if the immediately preceding pickup has been completed (lines 305-306).

### 3. Depot Masking (lines 313-323)

The depot can be visited under specific conditions to prevent infinite loops:

```python
depot_feasible = (i > 0) & load_is_zero & not_at_depot
```

- **Not first step**: `i > 0` (prevents immediate return)
- **Load is zero**: All picked-up passengers have been dropped off
- **Not at depot**: Prevents self-loops at depot

### 4. Emergency Depot Exit (lines 325-332)

Special fallback mechanism for infeasible instances:

```python
depot_emergency = no_actions_available & (i > 0)
action_mask[..., 0] = action_mask[..., 0] | depot_emergency
```

- **Purpose**: Ensures at least one valid action exists
- **Trigger**: No other feasible actions AND not first step
- **Effect**: Allows depot even if normally not feasible (including self-loops)
- **Handling**: Environment marks episode as `done` in `_step()` when this occurs (lines 154-158)

## State Variables Used

From TensorDict (lines 262-273):

- `visited`: Boolean array tracking visited nodes
- `picked_up`: Boolean array tracking pickups by current vehicle
- `current_node`: Current position
- `current_agent`: Active vehicle index
- `current_time`: Current time for the vehicle
- `current_load`: Current passenger load
- `capacity`: Vehicle capacity
- `time_windows`: Deadline for each node
- `demand`: Demand at each node (positive for pickup, negative for dropoff)

## Time and Distance Calculations (lines 275-283)

```python
dist = (all_locs - curr_loc).norm(p=2, dim=-1)
travel_time = torch.round(dist / vehicle_speed).long()
arrival_time = current_time.unsqueeze(-1) + travel_time
```

All nodes are evaluated simultaneously for vectorized feasibility checking.

## Agent Switching Logic

When an agent returns to depot with zero load (lines 126-136 in `_step()`):

1. Switch to next agent if conditions met
2. Reset time, load, and position to depot
3. **Clear `picked_up` state** for the new agent (line 135)
4. This ensures pickup-dropoff pairing is per-vehicle

## Key Design Features

1. **Vectorized Operations**: All feasibility checks computed in parallel across all nodes
2. **Batch Support**: Handles arbitrary batch dimensions `[B]` or `[B, S]`
3. **Infeasibility Handling**: Emergency depot exit prevents crashes on impossible instances
4. **Vehicle-specific Tracking**: `picked_up` state resets per vehicle to enforce same-vehicle pairing
5. **Prevents Deadlocks**: Depot self-loop as last resort ensures environment never gets stuck

## Related Methods

- **`_step()` (lines 72-178)**: Updates state based on selected action
- **`_reset()` (lines 180-251)**: Initializes state and first action mask
- **`check_solution_validity()` (lines 351-434)**: Post-hoc validation of complete solutions
