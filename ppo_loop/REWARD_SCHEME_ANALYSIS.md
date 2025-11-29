# Reward Scheme Analysis: Absolute Cost vs Percentage Improvement

## Current Reward Scheme (Option 1)

### Implementation
```python
# dvrp_env.py:468
reward = old_cost - new_cost - patience_penalty
```

Where:
- `old_cost`: Total routing cost before adding current request
- `new_cost`: Total routing cost after adding current request
- `patience_penalty`: `(|pickup_shift| + |dropoff_shift|) × patience_factor`

### Characteristics

**Advantages:**
1. ✅ **Stationary reward scale**: Cost differences are relatively stable across the episode
2. ✅ **Simple and interpretable**: Direct km savings
3. ✅ **No dependency on external baseline**: Self-contained per episode
4. ✅ **Myron's reward shaping**: Encourages incremental cost reduction

**Disadvantages:**
1. ❌ **Non-normalized rewards**: Scale varies with problem size
2. ❌ **No explicit baseline comparison**: Doesn't directly optimize vs no-negotiation
3. ❌ **Magnitude inconsistency**: Early steps have small costs (~10-50 km), late steps have large costs (~200-500 km)
4. ❌ **Sparse learning signal**: Small cost differences may not provide strong gradients

### Reward Distribution Example

| Step | Old Cost | New Cost | Reward | Magnitude |
|------|----------|----------|--------|-----------|
| 1 | 0 | 12.5 | -12.5 | Small |
| 5 | 45.2 | 52.3 | -7.1 | Small |
| 10 | 125.8 | 138.4 | -12.6 | Medium |
| 20 | 312.5 | 325.1 | -12.6 | Medium |
| 30 | 487.3 | 495.2 | -7.9 | Small |

**Issue**: Reward scale is absolute (km), which grows with episode progress but is dominated by routing solver's decisions rather than negotiation quality.

---

## Proposed Reward Scheme (Option 2)

### Implementation
```python
# Proposed modification
reward = ((baseline_cost - agent_cost) / baseline_cost) * 100 - patience_penalty
```

Where:
- `baseline_cost`: Routing cost if agent always chose action 12 (no time shift)
- `agent_cost`: Routing cost with agent's chosen action
- Result: Percentage improvement vs non-negotiation baseline

### Characteristics

**Advantages:**
1. ✅ **Normalized rewards**: Always in percentage scale (-100% to +100%)
2. ✅ **Direct optimization target**: Explicitly optimizes what we care about (% improvement)
3. ✅ **Scale-invariant**: Same reward magnitude whether total cost is 100 km or 500 km
4. ✅ **Clearer learning signal**: "You saved 5% vs baseline" is more informative than "You saved 12 km"
5. ✅ **Better for value function**: Easier to learn state values when rewards are bounded

**Disadvantages:**
1. ❌ **Requires parallel baseline**: Need to run baseline environment in sync
2. ❌ **Non-Markovian?**: Reward depends on hypothetical baseline trajectory
3. ❌ **Division by zero risk**: If baseline_cost = 0 (shouldn't happen, but edge case)
4. ❌ **Potential training instability**: Early in training, agent might perform much worse than baseline (large negative %)

### Reward Distribution Example

| Step | Agent Cost | Baseline Cost | Reward | Interpretation |
|------|------------|---------------|--------|----------------|
| 1 | 12.5 | 14.2 | +11.97% | 12% improvement! |
| 5 | 52.3 | 51.8 | -0.97% | Slightly worse |
| 10 | 138.4 | 142.1 | +2.60% | 2.6% improvement |
| 20 | 325.1 | 331.5 | +1.93% | Nearly 2% better |
| 30 | 495.2 | 502.3 | +1.41% | Small improvement |

**Benefit**: Rewards are bounded and normalized, making them easier to learn.

---

## Comparative Analysis

### Learning Dynamics

**Current Scheme (Absolute):**
- Early steps: reward ≈ -10 to -20 km
- Late steps: reward ≈ -5 to -15 km
- PPO sees: "Adding requests always costs ~10 km"
- Agent learns: "Minimize cost increases"

**Proposed Scheme (Percentage):**
- Early steps: reward ≈ -5% to +10%
- Late steps: reward ≈ -2% to +5%
- PPO sees: "I'm 3% better than baseline"
- Agent learns: "Maximize % improvement vs doing nothing"

### Value Function Approximation

**Current Scheme:**
- Value function must predict: "Total absolute cost savings over episode"
- Value range: -500 to +100 km (unbounded, problem-dependent)
- Hard to normalize across different problem instances

**Proposed Scheme:**
- Value function must predict: "Total % improvement over episode"
- Value range: Bounded by episode length × max % per step
- Easier to generalize across problem sizes

### Policy Gradient Signal

**Current Scheme:**
```
Advantage = Actual_reward - Value_estimate
          = (old_cost - new_cost) - V(s)
```
- Advantage dominated by routing solver's cost function
- Weak signal for negotiation quality

**Proposed Scheme:**
```
Advantage = Actual_reward - Value_estimate
          = ((baseline_cost - agent_cost) / baseline_cost) - V(s)
```
- Advantage directly measures negotiation effectiveness
- Stronger signal: "Did my negotiation beat the baseline?"

---

## Implementation Difficulty Assessment

### Option 2 Implementation: **EASY** ✅

The baseline is **already being computed** in `meta_train.py:176-184`!

```python
# meta_train.py:175-184
# Already implemented!
baseline_vec_env = VectorizedDVRPEnv(
    num_envs=num_envs,
    num_customers=num_customers,
    max_vehicles=5,
    solver_time_limit=1,
    seed=seed,  # Same seed for fair comparison
    traveler_decisions_path=traveler_decisions_path,
    device=device,
)
```

### Changes Required

**1. Modify `step()` to accept baseline_cost (Optional):**

```python
# dvrp_env.py or vectorized_env.py
def step(self, action: int, baseline_cost: Optional[float] = None) -> Tuple[...]:
    # ... existing code ...

    # NEW: Calculate reward based on scheme
    if baseline_cost is not None:
        # Percentage improvement scheme
        if baseline_cost > 0:
            improvement_pct = ((baseline_cost - new_cost) / baseline_cost) * 100
            reward = improvement_pct - patience_penalty
        else:
            # Fallback to absolute if baseline is invalid
            reward = self.previous_cost - new_cost - patience_penalty
    else:
        # Original absolute cost scheme
        reward = self.previous_cost - new_cost - patience_penalty

    # ... rest of code ...
```

**2. Modify `meta_train.py` to pass baseline costs:**

```python
# meta_train.py:330-369
# Current:
next_states, rewards, dones, truncs, step_infos = vec_env.step(actions)

# New:
# Get baseline costs from baseline environment
baseline_next_states, _, baseline_dones, baseline_truncs, baseline_step_infos = baseline_vec_env.step(baseline_actions)

# Extract baseline costs
baseline_costs = np.array([info.get('current_cost', 0.0) for info in baseline_step_infos])

# Step with baseline costs
next_states, rewards, dones, truncs, step_infos = vec_env.step(actions, baseline_costs=baseline_costs)
```

**Total changes: ~20 lines of code!**

---

## Alternative: Hybrid Reward Scheme

Combine both approaches for best of both worlds:

```python
# Hybrid reward
absolute_reward = old_cost - new_cost
if baseline_cost is not None and baseline_cost > 0:
    improvement_pct = ((baseline_cost - new_cost) / baseline_cost) * 100
    # Weight both components
    reward = 0.3 * absolute_reward + 0.7 * improvement_pct - patience_penalty
else:
    reward = absolute_reward - patience_penalty
```

This:
- Keeps absolute cost signal (helps with absolute routing quality)
- Adds percentage signal (encourages beating baseline)
- Weights can be tuned via hyperparameter search

---

## Recommendation

### **I recommend Option 2 (Percentage Improvement)** for these reasons:

1. **Aligns with objective**: You explicitly track "% improvement vs baseline" in logging (meta_train.py:469) - why not optimize it directly?

2. **Better learning signal**: Normalized rewards → easier value function learning → faster convergence

3. **Already implemented baseline**: You're running `baseline_vec_env` anyway, just not using it for rewards

4. **Easy to implement**: ~20 lines of code changes, minimal risk

5. **Interpretable**: "Agent learns to beat no-negotiation by X%" is clearer than "Agent reduces cost by Y km"

### Implementation Complexity: **2/10** (Very Easy)

- Baseline environment already exists ✅
- Just need to pass `baseline_cost` to `step()` ✅
- Backward compatible (can keep both schemes with a flag) ✅

### Potential Improvements to Training:

- **Faster convergence**: Clearer learning signal
- **Better generalization**: Rewards are problem-size invariant
- **More stable**: Bounded reward range helps PPO
- **Direct alignment**: Optimizes the metric you care about

---

## Experimental Validation Plan

1. **Keep current scheme as baseline**
2. **Implement percentage scheme with feature flag**
3. **Run A/B test:**
   - Train for 100 epochs with each scheme
   - Compare: convergence speed, final performance, stability
4. **Metrics to track:**
   - Average % improvement vs baseline (both should optimize this now)
   - Training time to reach 5% improvement threshold
   - Variance in episode rewards
   - Value function estimation error

Would you like me to implement Option 2?
