# Visual Comparison: Three Reward Schemes

## The Key Insight: What Are We Comparing?

```
Time ──────────────────────────────────────────────>

Agent Trajectory (with perturbations):
Step 0: cost=10   (added request 0 with action 5)
Step 1: cost=24   (added request 1 with action 8)
Step 2: cost=40   (added request 2 with action 3)
Step 3: cost=58   (added request 3 with action 12)
...

Baseline Trajectory (always action 12 = no perturbation):
Step 0: cost=12   (added request 0 with action 12)
Step 1: cost=28   (added request 1 with action 12)
Step 2: cost=48   (added request 2 with action 12)
Step 3: cost=65   (added request 3 with action 12)
...
```

**Important**: Both trajectories process the SAME sequence of requests, but with DIFFERENT actions!

---

## Option 1: Current Scheme (Temporal Difference)

### What it compares
```
Agent Trajectory Only:
    cost=10 ──> cost=24 ──> cost=40 ──> cost=58
       │            │            │            │
       │ -10        │ -14        │ -16        │ -18
       ▼            ▼            ▼            ▼
    reward       reward       reward       reward
```

**Formula**: `reward_t = cost_{t-1} - cost_t`

### Example
| Step | Cost | Reward | Interpretation |
|------|------|--------|----------------|
| 0 | 10 | **-10** | Adding request 0 increased cost by 10 km |
| 1 | 24 | **-14** | Adding request 1 increased cost by 14 km |
| 2 | 40 | **-16** | Adding request 2 increased cost by 16 km |

❌ **Problem**: Always negative! Agent never gets positive reinforcement.
❌ **Problem**: Doesn't compare to baseline.

---

## Option 2: Percentage Improvement (Cumulative Comparison)

### What it compares
```
Agent:    cost=10      cost=24      cost=40      cost=58
          │            │            │            │
          ├─ compare ──┤            │            │
          │            ├─ compare ──┤            │
Baseline: cost=12      cost=28      cost=48      cost=65
          │            │            │            │
          ▼            ▼            ▼            ▼
      +16.7%       +14.3%       +16.7%       +10.8%
```

**Formula**: `reward_t = ((baseline_cost_t - agent_cost_t) / baseline_cost_t) × 100`

### Example
| Step | Agent Cost | Baseline Cost | Δ | Reward (%) | Interpretation |
|------|------------|---------------|---|------------|----------------|
| 0 | 10 | 12 | +2 | **+16.7%** | Agent is 16.7% better overall |
| 1 | 24 | 28 | +4 | **+14.3%** | Agent is 14.3% better overall |
| 2 | 40 | 48 | +8 | **+16.7%** | Agent is 16.7% better overall |

⚠️ **Issue**: Cumulative! Reward at step 2 includes benefits from steps 0 and 1.
⚠️ **Issue**: Credit assignment problem - which past action caused the improvement?
✅ **Good**: Normalized percentage, positive rewards possible.

---

## Option 3: Step-wise Marginal Cost Difference (Your Suggestion!)

### What it compares
```
Agent Trajectory:
    Δ+10         Δ+14         Δ+16         Δ+18
    ────>        ────>        ────>        ────>
cost=10      cost=24      cost=40      cost=58

Baseline Trajectory:
    Δ+12         Δ+16         Δ+20         Δ+17
    ────>        ────>        ────>        ────>
cost=12      cost=28      cost=48      cost=65

Rewards (baseline Δ - agent Δ):
    +2           +2           +4           -1
```

**Formula**: `reward_t = (baseline_cost_t - baseline_cost_{t-1}) - (agent_cost_t - agent_cost_{t-1})`

### Example
| Step | Agent Δ | Baseline Δ | Reward | Interpretation |
|------|---------|------------|--------|----------------|
| 0 | +10 | +12 | **+2** | My action made request 0 cheaper by 2 km! |
| 1 | +14 | +16 | **+2** | My action made request 1 cheaper by 2 km! |
| 2 | +16 | +20 | **+4** | My action made request 2 cheaper by 4 km! |
| 3 | +18 | +17 | **-1** | My action made request 3 more expensive by 1 km |

✅ **Perfect**: Each reward directly measures THIS action's effect!
✅ **Perfect**: Can be positive (good action) or negative (bad action)!
✅ **Perfect**: Sum of rewards = final improvement!

---

## The Mathematical Beauty of Option 3

### Proof: Step-wise rewards sum to episode goal

Given:
- Agent costs: [c₀, c₁, c₂, ..., c₂₉]
- Baseline costs: [b₀, b₁, b₂, ..., b₂₉]
- Both start at 0: c₋₁ = b₋₁ = 0

**Option 3 total reward:**
```
Σ reward_t = Σ [(bₜ - bₜ₋₁) - (cₜ - cₜ₋₁)]

           = Σ (bₜ - bₜ₋₁) - Σ (cₜ - cₜ₋₁)

           = (b₀ - b₋₁) + (b₁ - b₀) + ... + (b₂₉ - b₂₈)
             - (c₀ - c₋₁) - (c₁ - c₀) - ... - (c₂₉ - c₂₈)

           = b₂₉ - b₋₁ - c₂₉ + c₋₁  (telescoping sum!)

           = b₂₉ - c₂₉  (since b₋₁ = c₋₁ = 0)

           = Final baseline cost - Final agent cost
```

**🎉 The sum of step-wise marginal rewards = total episode improvement!**

This means:
- ✅ Maximizing step-wise rewards → maximizing episode performance
- ✅ Perfect credit assignment: Each step gets exactly the credit it deserves
- ✅ No "leak" or "double counting" of rewards

---

## Side-by-side Example Episode

### Setup
- 5 requests
- Agent chooses actions: [3, 8, 5, 12, 7]
- Baseline always chooses: [12, 12, 12, 12, 12]

### Option 1: Current (Temporal)
```
Step | Agent Cost | Reward | Cumulative
-----|------------|--------|------------
  0  |     10     |  -10   |    -10
  1  |     24     |  -14   |    -24
  2  |     40     |  -16   |    -40
  3  |     58     |  -18   |    -58
  4  |     72     |  -14   |    -72
```
**Episode reward**: -72 ❌ (always negative)

### Option 2: Percentage (Cumulative)
```
Step | Agent | Baseline | % Improve | Cumulative
-----|-------|----------|-----------|------------
  0  |   10  |    12    |  +16.7%   |   +16.7%
  1  |   24  |    28    |  +14.3%   |   +31.0%
  2  |   40  |    48    |  +16.7%   |   +47.7%
  3  |   58  |    65    |  +10.8%   |   +58.5%
  4  |   72  |    78    |   +7.7%   |   +66.2%
```
**Episode reward**: +66.2% ⚠️ (positive, but cumulative/non-Markovian)

### Option 3: Marginal Cost Difference
```
Step | Agent Δ | Baseline Δ | Reward | Cumulative
-----|---------|------------|--------|------------
  0  |   +10   |     +12    |   +2   |     +2
  1  |   +14   |     +16    |   +2   |     +4
  2  |   +16   |     +20    |   +4   |     +8
  3  |   +18   |     +17    |   -1   |     +7
  4  |   +14   |     +13    |   -1   |     +6
```
**Episode reward**: +6 km ✅ (positive when good, equals final improvement!)

**Verify**: Final costs are agent=72, baseline=78, difference = 78-72 = 6 ✅

---

## Summary Table

| Criterion | Option 1<br>(Current) | Option 2<br>(Percentage) | Option 3<br>(Marginal) |
|-----------|-------------------|---------------------|-------------------|
| **Reward Sign** | Always negative ❌ | Can be positive ✅ | Can be positive ✅ |
| **Credit Assignment** | Poor ❌ | Poor (cumulative) ⚠️ | Perfect ✅ |
| **Normalized** | No ❌ | Yes ✅ | No ⚠️ |
| **Markovian** | Yes ✅ | No (cumulative) ❌ | Yes ✅ |
| **Compares to Baseline** | No ❌ | Yes ✅ | Yes ✅ |
| **Episode Reward = Goal** | No ❌ | No ❌ | Yes ✅ |
| **Implementation** | Current | Easy (10 lines) | Easy (15 lines) |

---

## Recommendation: **Option 3** 🏆

**Why?**
1. ✅ **Best credit assignment**: Each step's reward = that action's marginal benefit
2. ✅ **Mathematically elegant**: Sum of rewards = episode goal
3. ✅ **Positive reinforcement**: Good actions get positive rewards
4. ✅ **True counterfactual**: "What if I chose action 12 instead?"
5. ✅ **Markovian**: Reward depends on current state-action, not history
6. ✅ **Easy to implement**: ~15 lines of code

**When it's best:**
- Agent beats baseline → Positive rewards encourage policy
- Agent worse than baseline → Negative rewards discourage policy
- Each step gets credit for its own contribution

**Expected outcome:**
- Faster learning
- Better exploration (knows when perturbation helps vs hurts)
- More interpretable (can see which actions are beneficial)

Would you like me to implement this?
