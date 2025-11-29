# Three Reward Scheme Options: Detailed Comparison

## Context from meta_train.py

Current setup (lines 289-334):
```python
# Reset both environments with SAME seed
states, infos = vec_env.reset()
baseline_states, _ = baseline_vec_env.reset()

for step in range(num_customers):
    # Agent chooses actions based on policy
    actions = agent.select_action_batch(states, masks, epsilon)

    # Agent environment steps
    next_states, rewards, dones, truncs, step_infos = vec_env.step(actions)

    # Baseline always chooses action 12 (no perturbation)
    baseline_actions = np.full(num_envs, 12, dtype=np.int64)
    baseline_next_states, _, baseline_dones, baseline_truncs, baseline_step_infos = baseline_vec_env.step(baseline_actions)
```

**Key observation**: Both environments process the SAME sequence of requests, but with DIFFERENT actions. After step 0, the trajectories diverge!

---

## Option 1: Current Scheme (Temporal Difference)

### Formula
```python
# dvrp_env.py:468
reward = previous_cost - new_cost - patience_penalty
```

### What it measures
"How much did the routing cost change when I added this request to my current solution?"

### Characteristics
- **Comparison**: Current state vs previous state (SAME trajectory)
- **Sign**: Always negative (adding requests always increases cost)
- **Magnitude**: Depends on marginal routing cost

### Example Episode
| Step | Previous Cost | New Cost | Reward | Interpretation |
|------|---------------|----------|--------|----------------|
| 0 | 0 | 12 | -12 | Adding first request costs 12 km |
| 1 | 12 | 25 | -13 | Adding second request costs 13 km |
| 2 | 25 | 42 | -17 | Adding third request costs 17 km |
| ... | ... | ... | ... | ... |
| 29 | 485 | 502 | -17 | Adding last request costs 17 km |

**Total episode reward**: Sum of all negative marginal costs ≈ -500 km

### Issues
❌ **Non-stationary reward scale**: Early requests (~-10 km) vs late requests (~-20 km)
❌ **Always negative**: Agent never gets positive reinforcement
❌ **Not comparing to baseline**: Doesn't explicitly optimize vs no-negotiation
❌ **Reward dominated by routing solver**: Small differences in perturbation have small impact on marginal cost

---

## Option 2: Percentage Improvement (Normalized Comparison)

### Formula
```python
reward = ((baseline_cost_t - agent_cost_t) / baseline_cost_t) * 100 - patience_penalty
```

Where:
- `baseline_cost_t`: Total cost in baseline trajectory at step t
- `agent_cost_t`: Total cost in agent trajectory at step t

### What it measures
"What percentage better is my total routing cost compared to baseline at this step?"

### Characteristics
- **Comparison**: Agent trajectory vs baseline trajectory (PARALLEL)
- **Sign**: Positive if better than baseline, negative if worse
- **Magnitude**: Percentage (bounded, scale-invariant)

### Example Episode
| Step | Agent Cost | Baseline Cost | Raw Diff | Reward (%) | Interpretation |
|------|------------|---------------|----------|------------|----------------|
| 0 | 10 | 12 | +2 | +16.7% | 16.7% better than baseline! |
| 1 | 24 | 28 | +4 | +14.3% | Still 14% better |
| 2 | 40 | 48 | +8 | +16.7% | Maintaining 16% advantage |
| ... | ... | ... | ... | ... | ... |
| 29 | 495 | 530 | +35 | +6.6% | Final: 6.6% better overall |

**Total episode reward**: Sum of percentages (can vary, but bounded)

### Important Note: CUMULATIVE Comparison
⚠️ This compares **total routing costs**, not **marginal costs per step**!
- Step 0: Compares routing of request 0
- Step 10: Compares routing of requests 0-10 with DIFFERENT action histories
- Step 29: Compares routing of all requests 0-29 with DIFFERENT trajectories

The reward at step t includes the cumulative effect of all previous decisions!

### Advantages
✅ **Normalized**: Percentage is scale-invariant
✅ **Positive reinforcement**: Can be positive when beating baseline
✅ **Direct optimization**: Explicitly maximizes % improvement
✅ **Bounded rewards**: Helps value function learning

### Disadvantages
❌ **Cumulative, not marginal**: Reward at step t depends on all previous steps
❌ **Credit assignment issue**: Hard to know if current action or past actions caused improvement
❌ **Non-Markovian**: Reward depends on entire trajectory history

---

## Option 3: Step-wise Marginal Cost Comparison (What You're Suggesting)

### Formula
```python
# Marginal cost in baseline trajectory
baseline_marginal_cost_t = baseline_cost_t - baseline_cost_{t-1}

# Marginal cost in agent trajectory
agent_marginal_cost_t = agent_cost_t - agent_cost_{t-1}

# Reward = How much better is my marginal cost?
reward = baseline_marginal_cost_t - agent_marginal_cost_t - patience_penalty
```

### What it measures
"Did my action on THIS request make the marginal routing cost lower than if I had chosen no perturbation?"

### Characteristics
- **Comparison**: Marginal cost of current step in both trajectories (PARALLEL)
- **Sign**: Positive if agent's marginal cost < baseline's marginal cost
- **Magnitude**: Absolute km difference (unbounded)

### Example Episode
| Step | Agent Δ | Baseline Δ | Reward | Interpretation |
|------|---------|------------|--------|----------------|
| 0 | +10 | +12 | +2 | My action made adding request cheaper by 2 km! |
| 1 | +14 | +16 | +2 | Again, 2 km savings on this request |
| 2 | +16 | +20 | +4 | 4 km savings on this request! |
| ... | ... | ... | ... | ... |
| 29 | +17 | +18 | +1 | 1 km savings on last request |

**Total episode reward**: Sum of marginal savings ≈ +50 km

### Advantages
✅ **Step-wise credit assignment**: Reward directly attributed to current action
✅ **Positive reinforcement**: Can be positive when action improves marginal cost
✅ **True counterfactual**: Compares "what if I chose action 12 instead?"
✅ **Better Markovian**: Reward more closely tied to current state-action pair

### Disadvantages
❌ **Trajectories diverged**: Baseline and agent have different routing problems at step t
❌ **Scale still non-normalized**: km differences vary with problem size
❌ **Noisy signal**: Marginal costs can be noisy due to routing solver stochasticity

### Implementation Complexity
**EASY**: Requires storing `previous_cost` for baseline environment

---

## Visual Comparison: What Each Reward Measures

### Option 1 (Current - Temporal)
```
Agent Trajectory:
Step 0: cost=10
        ↓ reward = 0 - 10 = -10
Step 1: cost=24
        ↓ reward = 10 - 24 = -14
Step 2: cost=40
        ↓ reward = 24 - 40 = -16
```
**Compares**: Current step to previous step (same trajectory)

---

### Option 2 (Percentage - Cumulative Parallel)
```
Agent:     cost=10  cost=24  cost=40
Baseline:  cost=12  cost=28  cost=48
           ↓        ↓        ↓
Reward:    +16.7%   +14.3%   +16.7%
```
**Compares**: Total agent cost to total baseline cost at each step

---

### Option 3 (Marginal - Step-wise Parallel)
```
Agent:     Δ+10    Δ+14     Δ+16
Baseline:  Δ+12    Δ+16     Δ+20
           ↓       ↓        ↓
Reward:    +2      +2       +4
```
**Compares**: Marginal cost increase in agent vs baseline at each step

---

## Which is Best?

### For Your Problem: **Option 3 (Marginal Cost Comparison)** 🏆

**Why?**

1. **Best credit assignment**: Directly attributes reward to the current action's effect on marginal routing cost

2. **True counterfactual reasoning**: Agent learns "Would choosing action 12 have been better/worse for THIS request?"

3. **Positive reinforcement**: Agent gets positive rewards when its perturbation reduces marginal cost vs no perturbation

4. **Markovian**: Reward depends primarily on current state and action, not entire history

5. **Aligns with episode objective**: If every step has positive marginal reward, final cost will be better than baseline

### Mathematical Justification

Let's say agent achieves costs [c₀, c₁, c₂, ..., c₂₉] and baseline achieves [b₀, b₁, b₂, ..., b₂₉].

**Option 2** (Percentage):
```
Total reward = Σ((bₜ - cₜ) / bₜ) × 100
```
- Heavily weights early steps (small denominators)
- Cumulative comparison

**Option 3** (Marginal):
```
Total reward = Σ((bₜ - bₜ₋₁) - (cₜ - cₜ₋₁))
             = Σ(bₜ - bₜ₋₁) - Σ(cₜ - cₜ₋₁)
             = (b₂₉ - b₀) - (c₂₉ - c₀)
             = b₂₉ - c₂₉  (since b₀ = c₀ = 0)
```
- **This sums to exactly the final cost difference!**
- Each step's reward contributes to the total episode goal

✅ **Option 3 has perfect credit assignment**: Maximizing step-wise marginal rewards = maximizing episode improvement!

---

## Implementation Comparison

### Option 2: Percentage (Normalized)
```python
# In meta_train.py
baseline_costs = np.array([info['current_cost'] for info in baseline_step_infos])
agent_costs = np.array([info['current_cost'] for info in step_infos])

# Pass to environment
rewards = ((baseline_costs - agent_costs) / baseline_costs) * 100 - patience_penalties
```

**Changes**: ~10 lines in meta_train.py, ~5 lines in vectorized_env.py

---

### Option 3: Marginal Cost Difference
```python
# In dvrp_env.py - store baseline's previous_cost
self.baseline_previous_cost = 0.0

# In meta_train.py - after stepping baseline
baseline_marginal_costs = np.array([
    info['current_cost'] - env.baseline_previous_cost
    for env, info in zip(baseline_vec_env.envs, baseline_step_infos)
])

# Step agent and get marginal costs
agent_marginal_costs = np.array([
    info['current_cost'] - env.previous_cost
    for env, info in zip(vec_env.envs, step_infos)
])

# Reward = baseline marginal - agent marginal
rewards = baseline_marginal_costs - agent_marginal_costs - patience_penalties

# Update baseline's previous_cost
for env, info in zip(baseline_vec_env.envs, baseline_step_infos):
    env.baseline_previous_cost = info['current_cost']
```

**Changes**: ~15 lines in meta_train.py, ~2 lines in dvrp_env.py

---

## My Recommendation: **Option 3** 🎯

**Reasons:**
1. ✅ **Perfect credit assignment**: Step rewards sum to episode goal
2. ✅ **True counterfactual**: Explicitly compares to "what if I didn't perturb?"
3. ✅ **Positive rewards**: Agent gets positive feedback for good actions
4. ✅ **Easy to implement**: ~20 lines of code
5. ✅ **Theoretically sound**: Maximizes exactly what we want

**Expected improvements:**
- 🚀 Faster learning (clearer signal per step)
- 🎯 Better policies (learns when to perturb vs not perturb)
- 📈 More stable training (positive/negative rewards balance)

Would you like me to implement Option 3?
