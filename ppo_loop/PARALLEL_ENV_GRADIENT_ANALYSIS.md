# How PPO Handles Parallel Environments: Gradient Computation

## Your Question

> "When there are multiple envs, are you simply seeing the averaged reward or did you consider the difference among different env when you calculate the gradient?"

## Short Answer

**No, we don't just average rewards!** ✅

PPO **computes advantages separately for each environment**, then mixes them during mini-batch training. Gradients fully account for per-environment differences.

---

## Detailed Explanation

### Step-by-Step: How Parallel Environments Work

#### 1. **Data Collection** (Interleaved Storage)

During rollout, data is stored in **interleaved** format:

```python
# ppo_agent.py:461-465
for i in range(batch_size):  # batch_size = num_envs
    self.states.append(state_cpu[i])
    self.actions.append(actions_cpu[i].item())
    self.log_probs.append(log_probs_cpu[i])
    self.values.append(values_cpu[i])

# ppo_agent.py:488-490
for reward, done in zip(rewards, dones):
    self.rewards.append(float(reward))
    self.dones.append(bool(done))
```

**Buffer structure** (example with 3 envs, 2 steps):
```
states:   [s0_env0, s0_env1, s0_env2, s1_env0, s1_env1, s1_env2]
rewards:  [r0_env0, r0_env1, r0_env2, r1_env0, r1_env1, r1_env2]
actions:  [a0_env0, a0_env1, a0_env2, a1_env0, a1_env1, a1_env2]
```

**Key insight**: Data from different environments is interleaved, NOT averaged!

---

#### 2. **Advantage Computation** (Per-Environment GAE)

When updating, PPO uses `_compute_gae_parallel()` which:

**A. Reshapes data to separate trajectories:**

```python
# ppo_agent.py:835-837
# From: [env0_t0, env1_t0, env2_t0, env0_t1, env1_t1, env2_t1, ...]
# To:   [[env0_t0, env0_t1, env0_t2, ...],  # Environment 0
#        [env1_t0, env1_t1, env1_t2, ...],  # Environment 1
#        [env2_t0, env2_t1, env2_t2, ...]]  # Environment 2

rewards_2d = rewards.reshape(num_steps, num_envs).T  # (num_envs, num_steps)
values_2d = values.reshape(num_steps, num_envs).T
dones_2d = dones.reshape(num_steps, num_envs).T
```

**B. Computes GAE separately for EACH environment:**

```python
# ppo_agent.py:842-863
for env_idx in range(num_envs):  # ← Loop over each environment!
    last_advantage = 0.0

    for t in reversed(range(num_steps)):
        # Terminal state check for THIS environment
        if t == num_steps - 1 or dones_2d[env_idx, t]:
            next_value = 0.0
        else:
            next_value = values_2d[env_idx, t + 1]  # ← Same env, next step

        # TD error for THIS environment
        delta = (
            rewards_2d[env_idx, t]  # ← THIS env's reward
            + self.gamma * next_value * (1 - dones_2d[env_idx, t])
            - values_2d[env_idx, t]
        )

        # GAE for THIS environment
        advantages_2d[env_idx, t] = last_advantage = (
            delta
            + self.gamma * self.gae_lambda * (1 - dones_2d[env_idx, t]) * last_advantage
        )
```

**Result**: Each environment gets its own advantage calculation!

---

#### 3. **Mini-Batch Training** (Mix and Match)

After computing per-environment advantages, they're flattened and shuffled:

```python
# ppo_agent.py:867-868
# Flatten back to interleaved format
advantages = advantages_2d.T.reshape(-1)  # (num_envs * num_steps,)
returns = advantages + values

# ppo_agent.py:689
# Normalize advantages across ALL environments
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# ppo_agent.py:702-711
# Shuffle and create mini-batches
idx = np.random.permutation(num_samples)  # ← Shuffle ALL samples

for start in range(0, num_samples, batch_size):
    batch_idx = idx[start:end]  # Random samples from all environments
    batch_states = states[batch_idx]
    batch_actions = actions[batch_idx]
    batch_advantages = advantages[batch_idx]  # ← Mix of different envs
```

**Key point**: Mini-batches contain random samples from ALL environments!

---

#### 4. **Gradient Computation** (Per-Sample)

Gradients are computed per sample in the mini-batch:

```python
# ppo_agent.py:714-726
log_probs, _, entropy = self.policy.evaluate(batch_states, batch_actions)

# PPO ratio (per sample in batch)
ratio = torch.exp(log_probs - batch_old_log_probs)

# PPO objective (per sample in batch)
surr1 = ratio * batch_advantages  # ← Each sample has its own advantage
surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

# Average loss over mini-batch (gradient flows through each sample)
policy_loss = -torch.min(surr1, surr2).mean()
```

**Gradients account for**:
- Each environment's unique trajectory
- Each environment's unique rewards
- Each environment's unique advantages

---

## Visual Example

### Scenario: 2 environments, 3 steps

**Environment 0** (performs well):
```
Step | Reward | Value | TD Error | Advantage
-----|--------|-------|----------|----------
  0  |  +5    |  2.0  |   +3.5   |   +8.2
  1  |  +4    |  3.0  |   +2.5   |   +4.7
  2  |  +3    |  2.5  |   +1.5   |   +2.2
```

**Environment 1** (performs poorly):
```
Step | Reward | Value | TD Error | Advantage
-----|--------|-------|----------|----------
  0  |  -8    |  2.0  |  -10.0   |  -22.5
  1  |  -7    |  3.0  |  -11.0   |  -12.5
  2  |  -5    |  2.5  |   -8.5   |   -1.5
```

**Interleaved buffer**:
```
states:   [s0_e0, s0_e1, s1_e0, s1_e1, s2_e0, s2_e1]
rewards:  [+5,    -8,    +4,    -7,    +3,    -5   ]
advantages: [+8.2, -22.5, +4.7, -12.5, +2.2,  -1.5 ]  ← NOT averaged!
```

**Mini-batch** (batch_size=3, random shuffle):
```
Batch 1: [s1_e0, s0_e1, s2_e1]  → advantages: [+4.7, -22.5, -1.5]
Batch 2: [s2_e0, s1_e1, s0_e0]  → advantages: [+2.2, -12.5, +8.2]
```

**Gradient computation**:
```python
# For Batch 1:
policy_loss = -mean(
    min(ratio[0] * 4.7,  clipped_ratio[0] * 4.7),   # From env 0
    min(ratio[1] * -22.5, clipped_ratio[1] * -22.5), # From env 1
    min(ratio[2] * -1.5,  clipped_ratio[2] * -1.5)   # From env 1
)
```

**Each sample contributes to the gradient based on its own advantage!**

---

## What Gets Averaged? What Doesn't?

### ✅ Computed Per-Environment (NOT Averaged)

| Quantity | Computed How |
|----------|--------------|
| **Advantages** | Per-environment GAE (separate TD errors) |
| **Returns** | Per-environment (advantages + values) |
| **TD errors** | Per-environment (own rewards, values, dones) |
| **Value targets** | Per-environment trajectories |

### ⚠️ Normalized/Averaged Across Environments

| Quantity | Why |
|----------|-----|
| **Advantage normalization** | `(A - mean(A)) / std(A)` across all environments (lines 689) |
| **Loss** | Mean over mini-batch (contains mixed environments) (line 723) |

**Important**: Normalization ≠ Losing information!
- Normalization just scales advantages to have mean=0, std=1
- Relative differences between environments are preserved
- Gradient still flows based on per-sample advantages

---

## Does Averaging Rewards Lose Information?

### No! Here's why:

**Scenario**: Two environments with different reward scales
- Env 0: All rewards ≈ +10
- Env 1: All rewards ≈ -10

**What happens**:
1. **GAE computes advantages per-env** → Env 0 gets positive advantages, Env 1 gets negative
2. **Normalization** → Scales to mean=0, std=1 (but preserves ranking)
3. **Gradient** → Policy pushed toward actions from Env 0, away from actions from Env 1

**Result**: Policy learns from the difference between good and bad environments!

---

## Summary Table

| Stage | Per-Environment? | Details |
|-------|------------------|---------|
| **Data collection** | ✅ Yes | Each env stores own trajectory |
| **Advantage computation** | ✅ Yes | Separate GAE per environment |
| **Normalization** | ❌ No | Across all environments |
| **Mini-batch creation** | ❌ No | Random shuffle from all envs |
| **Gradient computation** | ✅ Yes | Per-sample in batch (from different envs) |

---

## Conclusion

**Your concern**: Are we just averaging rewards and losing per-environment information?

**Answer**: **No!** ✅

1. ✅ **Advantages are computed per-environment** (separate GAE loops)
2. ✅ **Each environment's trajectory is treated independently** during GAE
3. ✅ **Gradients account for per-environment differences** (per-sample advantages)
4. ⚠️ **Normalization happens AFTER** per-env computation (preserves relative differences)
5. ✅ **Mini-batches mix environments** (good for variance reduction in gradients)

**The implementation is correct!** Parallel environments provide:
- More diverse training data
- Better exploration
- Variance reduction in gradient estimates
- **WITHOUT** losing per-environment information

🎯 **Each environment's unique experience contributes to the gradient based on its own advantages!**
