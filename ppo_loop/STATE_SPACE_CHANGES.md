# State Space Redesign and Information Flow

**Date:** 2025-11-28
**Version:** 2.0

---

## Table of Contents

1. [Overview](#overview)
2. [State Space Changes](#state-space-changes)
3. [Information Flow](#information-flow)
4. [File Modifications](#file-modifications)
5. [Time Window Tracking](#time-window-tracking)
6. [Usage Examples](#usage-examples)

---

## Overview

This document describes the major redesign of the state space representation for the DVRP-TW environment, transitioning from dense scalar features to sparse one-hot encoded representations.

### Key Changes

- **State representation**: Dense (30, 6) → Sparse one-hot (606, 2)
- **Encoding**: Location indices and time windows → One-hot vectors
- **Dual tracking**: Maintains both perturbed (for routing) and real (for oracle) requests
- **Dynamic sizing**: Automatically adapts to number of distinct locations in dataset

---

## State Space Changes

### Old State Space (v1.0)

**Shape:** `(30, 6)`

```
Row structure (per request):
[pickup_h3_idx, pickup_tw_early, pickup_tw_late,
 dropoff_h3_idx, dropoff_tw_early, dropoff_tw_late]
```

**Properties:**
- Fixed 30 rows (one per customer)
- Dense scalar representation
- Location as H3 index (integer)
- Time windows as continuous values (minutes)

### New State Space (v2.0)

**Shape:** `(state_rows, 2)` where `state_rows = 2 × num_distinct_locations + 192`

For SF dataset with 207 distinct locations: `(606, 2)`

**Column Structure:**
- **Column 0**: Aggregated sum of all previously accepted requests (one-hot encodings)
- **Column 1**: Current new incoming request (one-hot encoding)

**Row Structure:**

```
Rows [0:207]         : Pickup location one-hot (207 locations)
Rows [207:414]       : Dropoff location one-hot (207 locations)
Rows [414:462]       : Pickup TW early (48 time intervals × 30 min)
Rows [462:510]       : Pickup TW late (48 time intervals × 30 min)
Rows [510:558]       : Dropoff TW early (48 time intervals × 30 min)
Rows [558:606]       : Dropoff TW late (48 time intervals × 30 min)
```

**Properties:**
- Sparse one-hot representation
- Each request contributes exactly 6 "hot" values (2 locations + 4 time windows)
- Column 0 accumulates all accepted requests via summation
- Column 1 always shows the current new request
- Time encoded in 30-minute intervals (0-1440 minutes → 48 intervals)

---

## Information Flow

### 1. Data Generation (SF_Generator)

```
┌─────────────────────────────────────────────────────────┐
│ SF_Generator.reset()                                    │
│                                                         │
│ Creates pending_requests TensorDict:                   │
│ ├─ h3_indices: Location indices                        │
│ ├─ time_windows: PRE-PERTURBED by generator (random)   │
│ ├─ user_id: Traveler IDs                              │
│ └─ trip_metadata: Original data (strings)              │
│    ├─ departure_time_window: "08:00-08:30" (string)   │
│    ├─ arrival_time_window: "09:00-09:30" (string)     │
│    ├─ flexibility: "flexible for both..."             │
│    ├─ trip_purpose: "work", "leisure", etc.           │
│    └─ locations: "Home", "Work", etc.                  │
└─────────────────────────────────────────────────────────┘
```

**Note:** The generator's perturbation is for **data augmentation only** and does not affect acceptance logic.

### 2. Environment Reset

```
DVRPEnv.reset()
├─ pending_requests = data_generator.reset()
├─ current_requests = depot only (for routing solver)
├─ real_requests = depot only (for oracle evaluation)
└─ Returns: observation (606, 2) with Column 0=zeros, Column 1=first request
```

### 3. Agent Observation

```
State Encoding (_get_observation):

For each accepted request in current_requests:
  ├─ Extract: h3_pickup, h3_dropoff, 4 time windows (perturbed)
  ├─ Encode to one-hot vector (606 dimensions)
  └─ Add to Column 0 (accumulation)

For current new request in pending_requests[current_step]:
  ├─ Extract: h3_pickup, h3_dropoff, 4 time windows (pre-perturbed)
  ├─ Encode to one-hot vector (606 dimensions)
  └─ Set as Column 1

Return: (606, 2) numpy array
```

### 4. Agent Decision

```
PPOAgent.select_action(state)
├─ Flatten: (606, 2) → (1212,)
├─ PolicyNetwork(1212) → action_logits (16)
├─ Apply mask (based on predicted flexibility)
├─ Sample action from Categorical distribution
└─ Return: action ∈ [0, 15]

Action Mapping:
  Action i → (pickup_shift, dropoff_shift)
  where pickup_shift ∈ {-30, -20, -10, 0}
        dropoff_shift ∈ {0, +10, +20, +30}
```

### 5. Acceptance Decision

```
DVRPEnv.step(action)
├─ Decode action: (pickup_shift, dropoff_shift)
├─ Get traveler metadata from pending_requests["trip_metadata"]
├─ Lookup acceptance in CSV:
│  ├─ Match: (traveler_id, trip_purpose, locations, TW_strings, flexibility)
│  ├─ With: (pickup_shift_abs, dropoff_shift_abs)
│  └─ Return: "accept" or "reject"
└─ Decision: accepted = (CSV says "accept")
```

**Key Insight:** Acceptance is based on:
- Categorical features from `trip_metadata` (strings)
- Agent's action (shift amounts)
- **NOT** the numerical time window values from tensors

### 6. Request Processing

```
If accepted:
  ├─ perturbed_request = apply_perturbation(new_request, action)
  │  └─ TW_perturbed = TW_pre-perturbed + action_shift
  └─ patience_penalty = (|pickup_shift| + |dropoff_shift|) × 0.2
Else:
  ├─ perturbed_request = new_request.clone()
  └─ patience_penalty = 0

# Build REAL request (original TW for oracle)
real_request = new_request.clone()
├─ Parse TW strings from trip_metadata:
│  ├─ "08:00-08:30" → (480, 510) minutes
│  └─ "09:00-09:30" → (540, 570) minutes
└─ Set real_request["time_windows"] = original_TW

# Update tracking
current_requests ← append(perturbed_request)  # For solver
real_requests ← append(real_request)          # For oracle
```

### 7. Routing Optimization

```
Solve routing with current_requests:
├─ If using OR-Tools:
│  └─ darp_solver(current_requests) → total_time
├─ If using neural policy:
│  └─ AttentionModel(current_requests) → -reward
└─ new_cost = routing cost
```

### 8. Reward Calculation

```
reward = old_cost - new_cost - patience_penalty

Where:
├─ old_cost: Previous routing cost
├─ new_cost: Current routing cost (from solver)
└─ patience_penalty: User inconvenience from time shifts
```

### 9. Next Observation

```
_get_observation() builds next state:
├─ Column 0: Sum of all accepted requests (current_requests, perturbed)
└─ Column 1: Next new request (pending_requests[step+1], pre-perturbed)
```

---

## File Modifications

### `dvrp_env.py`

#### New Attributes

```python
self.num_distinct_locations    # Read from travel_time_matrix.shape[0]
self.num_time_intervals = 48   # Day divided into 48×30min intervals
self.state_rows                # 2*num_locations + 192

# Dual tracking
self.current_requests  # Perturbed TWs for routing solver
self.real_requests     # Original TWs for oracle evaluation
```

#### New Methods

```python
_encode_location_onehot(location_idx) → np.ndarray
  # Converts location index to one-hot vector (num_distinct_locations,)

_encode_time_onehot(time_value) → np.ndarray
  # Converts time (minutes) to one-hot vector (48,)
  # Example: 320 min → interval 10 (320 // 30 = 10)

_encode_request_onehot(pickup_loc, dropoff_loc, 4_time_windows) → np.ndarray
  # Combines all features into one-hot vector (state_rows,)

_parse_time_window_string(tw_string) → (early_min, late_min)
  # Parses "08:00-08:30" → (480, 510)

get_real_requests() → TensorDict
  # Public method for oracle to access original time windows
```

#### Modified Methods

```python
__init__():
  # Dynamically reads num_distinct_locations
  # Sets observation_space to (state_rows, 2)

reset():
  # Initializes both current_requests and real_requests

step(action):
  # Builds perturbed_request (for solver)
  # Builds real_request (original TW from metadata)
  # Updates both tracking structures

_get_observation():
  # Column 0: Aggregates current_requests (perturbed)
  # Column 1: Shows new request from pending_requests (pre-perturbed)
```

### `ppo_agent.py`

#### Updated Network Dimensions

```python
PolicyNetwork:
  state_dim: 180 → 1212  # Default for SF dataset

ValueNetwork:
  state_dim: 180 → 1212

ActorCritic:
  state_dim: 180 → 1212
  forward(): Flattens (batch, 606, 2) → (batch, 1212)
```

#### Fixed Methods

```python
_compute_gae():
  # Added tensor flattening to handle 0-dim tensors
  # Prevents indexing errors with small batches
```

### `meta_train.py`

#### Dynamic State Dimension

```python
# Old (hardcoded)
state_dim = num_customers * 6  # 30 * 6 = 180

# New (dynamic)
obs_shape = vec_env.envs[0].observation_space.shape  # (606, 2)
state_dim = obs_shape[0] * obs_shape[1]              # 1212
```

---

## Time Window Tracking

### Three Types of Time Windows

| Type | Source | Purpose | Format | Perturbed? |
|------|--------|---------|--------|------------|
| **Original** | `trip_metadata["departure_time_window"]` | Acceptance lookup, Oracle evaluation | String: "08:00-08:30" | No |
| **Pre-perturbed** | `pending_requests["time_windows"]` | Data augmentation, Initial state | Tensor (minutes) | Yes (by SF_Generator) |
| **Perturbed** | `current_requests["time_windows"]` | Routing optimization | Tensor (minutes) | Yes (Generator + Agent) |

### Example Flow

```
Original TW (metadata):     "08:00-08:30"  = [480, 510] min
                                 ↓
        SF_Generator perturbation (random: +10 min)
                                 ↓
Pre-perturbed TW (pending): [490, 520] min  (shown in state Column 1)
                                 ↓
              Agent action (-30, 0)
                                 ↓
Perturbed TW (current):     [460, 490] min  (used by solver, shown in state Column 0)


Oracle compares:
├─ Baseline: Solve with Original TW [480, 510]
└─ Agent:    Solve with Perturbed TW [460, 490]
```

---

## Usage Examples

### Training with New State Space

```python
from dvrp_env import DVRPEnv
from ppo_agent import PPOAgent

# Environment automatically configures state space
env = DVRPEnv(num_customers=30)
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
# state_dim = 1212 for SF dataset

# Agent adapts to state dimension
agent = PPOAgent(state_dim=state_dim, action_dim=16)

# Training loop
obs, _ = env.reset()  # obs.shape = (606, 2)
action = agent.select_action(obs)  # Automatically flattens to (1212,)
next_obs, reward, done, _, info = env.step(action)
```

### Oracle Evaluation

```python
# Get requests with original time windows
real_requests = env.get_real_requests()

# Compare baseline vs agent
baseline_cost = solve_routing(real_requests)  # Original TWs
agent_cost = solve_routing(env.current_requests)  # Perturbed TWs

improvement = (baseline_cost - agent_cost) / baseline_cost * 100
print(f"Agent improved routing by {improvement:.1f}%")
```

### State Inspection

```python
obs, _ = env.reset()

# Column 0: Previously accepted requests (starts empty)
print(f"Aggregated requests: {obs[:, 0].sum()}")  # 0.0 initially

# Column 1: Current new request (always 6 hot values)
print(f"New request encoding: {obs[:, 1].sum()}")  # 6.0

# After accepting a request
obs, _, _, _, _ = env.step(action)
print(f"Aggregated requests: {obs[:, 0].sum()}")  # 6.0 (one request)
print(f"New request encoding: {obs[:, 1].sum()}")  # 6.0 (next request)
```

---

## Benefits of New Design

### 1. **Sparse Representation**
- More efficient for neural networks (most values are 0)
- Clearer semantic meaning (one-hot = exact feature value)

### 2. **Scalability**
- Automatically adapts to different numbers of locations
- No hardcoded dimension assumptions

### 3. **Interpretability**
- Column 0 clearly shows cumulative accepted requests
- Column 1 clearly shows current decision point
- Easy to track what the agent has accepted

### 4. **Oracle Integration**
- Separate `real_requests` for ground truth evaluation
- Compare perturbed routing vs original time windows
- Accurate performance metrics

### 5. **Information Preservation**
- Column 0 retains full history through summation
- No information loss from padding/truncation
- Agent sees all previous decisions

---

## Key Insights

### State Transition Dynamics

1. **Column 1** always shows the "raw" new request (pre-perturbed by generator)
2. **Agent action** determines additional perturbation
3. **Acceptance** determines if request is added to Column 0
4. **Column 0** accumulates with perturbations applied
5. **Oracle** evaluates using original (metadata) time windows

### Acceptance Logic

- **Input**: Categorical features (strings) + action shifts
- **Lookup**: CSV database of traveler decisions
- **Output**: Binary accept/reject
- **Independent** of numerical TW values in tensors

### Reward Semantics

- **Routing improvement**: (old_cost - new_cost)
- **User inconvenience**: patience_penalty
- **Trade-off**: Better routes vs user satisfaction

---

## Version History

- **v1.0** (Before 2025-11-28): Dense scalar state (30, 6)
- **v2.0** (2025-11-28): One-hot encoded state (606, 2), dual request tracking

---

## References

- `dvrp_env.py:32-61` - DVRPEnv class documentation
- `dvrp_env.py:438-522` - One-hot encoding methods
- `dvrp_env.py:867-875` - Oracle access method
- `ppo_agent.py:18-61` - PolicyNetwork architecture
- `meta_train.py:186-205` - Dynamic state_dim configuration
