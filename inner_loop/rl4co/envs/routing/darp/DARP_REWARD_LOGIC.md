# DARP Reward Calculation Logic

## Overview

The DARP environment uses a **sparse reward** scheme where rewards are only computed at episode termination. The reward function balances two objectives: minimizing travel distance and maximizing the number of customers served.

## Reward Formula

```python
# From _get_reward() at lines 340-348
num_unvisited = (~td["visited"][..., 1:]).sum(dim=-1).float()
cost = td["total_distance"] + penalty_unvisited * num_unvisited
reward = -cost
```

**Mathematical formulation:**

```
reward = -(total_distance + penalty_unvisited × num_unvisited)
```

Where:
- **`total_distance`**: Cumulative Euclidean distance traveled by all vehicles
- **`penalty_unvisited`**: Penalty coefficient per unvisited customer (default: **100.0**)
- **`num_unvisited`**: Count of customer nodes not visited (excludes depot)

## Components Breakdown

### 1. Total Distance Tracking

**Initialization** (line 203 in `_reset()`):
```python
total_distance = torch.zeros(*batch_size, dtype=torch.float32, device=device)
```

**Accumulation** (lines 87-89, 115 in `_step()`):
```python
# Calculate Euclidean distance from previous to current node
prev_loc = gather_by_index(td["locs"], td["current_node"])
curr_loc = gather_by_index(td["locs"], current_node)
dist = torch.norm(curr_loc - prev_loc, p=2, dim=-1, keepdim=True)

# Accumulate total distance
total_distance = td["total_distance"] + dist.squeeze(-1)
```

**Key points:**
- Uses **L2 norm (Euclidean distance)** in 2D coordinate space
- Distance accumulated **at every step**, including:
  - Depot → first customer
  - Customer → customer
  - Customer → depot
  - Depot → depot (emergency self-loop, though rare)
- Accumulated across **all vehicles** (not reset when switching agents)

### 2. Unvisited Node Penalty

**Calculation** (line 343):
```python
num_unvisited = (~td["visited"][..., 1:]).sum(dim=-1).float()
```

**Breakdown:**
- `td["visited"]`: Boolean tensor `[..., num_loc + 1]` tracking all nodes
- `[..., 1:]`: Slice to exclude depot (index 0)
- `~td["visited"][..., 1:]`: Invert to get unvisited customers
- `.sum(dim=-1)`: Count unvisited nodes per batch instance

**Penalty coefficient** (line 62):
- Default: `penalty_unvisited = 100.0`
- Configurable via constructor parameter
- Acts as a **heavy penalty** to discourage partial solutions

### 3. Cost Composition (line 346)

```python
cost = td["total_distance"] + self.penalty_unvisited * num_unvisited
```

**Example calculations:**

| Scenario | Distance | Unvisited | Penalty Coeff | Cost | Reward |
|----------|----------|-----------|---------------|------|--------|
| Perfect solution | 50.0 | 0 | 100.0 | 50.0 | **-50.0** |
| 2 nodes skipped | 45.0 | 2 | 100.0 | 245.0 | **-245.0** |
| Agent got stuck | 30.0 | 8 | 100.0 | 830.0 | **-830.0** |
| Emergency exit | 20.0 | 10 | 100.0 | 1020.0 | **-1020.0** |

The penalty dominates when customers are unvisited, creating strong incentive for complete solutions.

## Reward Timing: Sparse Rewards

### During Episode (line 160-161 in `_step()`):
```python
# Reward is 0 during episode, computed at the end via get_reward
reward = torch.zeros_like(done, dtype=torch.float32)
```

- **All intermediate steps return `reward = 0`**
- No reward shaping or step-by-step feedback
- Agent must learn from delayed terminal reward

### At Episode Termination:
- `_get_reward()` is called by the RL algorithm when episode ends
- Final reward computed based on complete trajectory
- Used for policy gradient updates (REINFORCE, PPO, etc.)

## Reward Characteristics

### 1. **Always Negative**
- Reward is always ≤ 0 (minimization formulation)
- Better solutions have **higher (less negative) rewards**
- Best possible: `-total_distance` when all customers served

### 2. **Multi-Objective Trade-off**
- **Short routes**: Lower distance, but might skip customers
- **Complete coverage**: Visit all customers, but longer routes
- Penalty coefficient (100.0) strongly weights completion over distance

### 3. **Non-Differentiable**
- Euclidean distance computation is differentiable
- But action selection is discrete (not differentiable w.r.t. policy parameters)
- Requires policy gradient methods (REINFORCE, PPO, A2C)

### 4. **Batch-Compatible**
- All operations vectorized across batch dimension
- Supports arbitrary batch shapes `[B]` or `[B, S]`

## Impact on Learning

### Strong Incentives Created:

1. **Visit all customers**: Each unvisited node costs 100 distance units
   - Equivalent to ~2-4 typical edges in unit square
   - Makes partial solutions highly undesirable

2. **Avoid getting stuck**: Emergency exits incur massive penalties
   - Encourages learning feasible action sequences
   - Punishes time window violations and poor planning

3. **Minimize route length**: Among complete solutions, shorter is better
   - Drives efficient routing decisions
   - Encourages good pickup-dropoff pairing

### Potential Issues:

1. **Sparse reward problem**: No intermediate feedback
   - Can slow initial learning
   - Might benefit from reward shaping (not currently implemented)

2. **Penalty sensitivity**: Fixed penalty (100.0) might not scale well
   - Too high: Agent overfits to completion, ignores distance
   - Too low: Agent might skip hard-to-reach customers
   - Should potentially scale with problem size

3. **No time window cost**: Time constraints only matter via feasibility
   - Tight schedules not rewarded over loose ones
   - Only enforced through action masking

## Comparison with Related Problems

| Problem | Reward Function | Key Difference |
|---------|----------------|----------------|
| **TSP** | `-total_distance` | No penalty needed (tour always complete) |
| **CVRP** | `-total_distance` | No penalty (all customers must be served) |
| **CVRPTW** | `-total_distance` or `-(dist + penalty)` | Similar penalty structure for infeasible instances |
| **DARP** | `-(dist + 100 × unvisited)` | Heavy penalty for partial solutions due to complexity |

## Customization

To modify reward behavior, adjust in `__init__`:

```python
env = DARPEnv(
    generator_params={...},
    penalty_unvisited=50.0,  # Lighter penalty
)
```

Or implement custom reward in subclass:
```python
class CustomDARPEnv(DARPEnv):
    def _get_reward(self, td, actions):
        # Custom logic here
        # E.g., add time window penalties, vehicle count penalties, etc.
        return custom_reward
```

## Related Code Locations

- **Reward calculation**: `_get_reward()` at lines 340-348
- **Distance accumulation**: `_step()` at lines 87-89, 115
- **Distance initialization**: `_reset()` at line 203
- **Penalty parameter**: `__init__()` at line 62
- **Intermediate rewards**: `_step()` at lines 160-161 (always zero)
