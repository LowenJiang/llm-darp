# Per-Epoch Evaluation Implementation Plan

## Goal

Add evaluation after each training epoch to monitor agent performance on **fixed test environments**.

---

## Requirements

1. ✅ **8 parallel fixed environments** (same seed for reproducibility)
2. ✅ **Greedy policy rollout** (no epsilon exploration)
3. ✅ **Compute metrics**: avg improvement %, avg routing cost
4. ✅ **Log to wandb** after each epoch
5. ✅ **Minimal overhead** (fast evaluation)

---

## Architecture Design

### 1. **Fixed Evaluation Environments**

**Why fixed?**
- Same environments every epoch → track progress over time
- No randomness → clearer learning signal
- Consistent baseline for comparison

**Setup**:
```python
# Create ONCE at start of training
eval_agent_env = VectorizedDVRPEnv(
    num_envs=8,
    num_customers=30,
    seed=9999,  # Fixed seed!
    ...
)

eval_baseline_env = VectorizedDVRPEnv(
    num_envs=8,
    num_customers=30,
    seed=9999,  # Same seed as agent env
    ...
)
```

**Seed strategy**:
- Use `seed=9999` (different from training seed=42)
- This ensures eval episodes are unseen during training
- 8 envs with base_seed=9999 → envs get seeds [9999, 10000, 10001, ..., 10006]

---

### 2. **Greedy Rollout**

**Current `select_action_batch()` has epsilon parameter**:
```python
# Training (with exploration)
actions = agent.select_action_batch(states, masks, epsilon=0.2)

# Evaluation (greedy, no exploration)
actions = agent.select_action_batch(states, masks, epsilon=0.0)
```

**Also need to handle masking**:
- Get predicted flexibility for each customer
- Compute masks based on embeddings
- Pass to select_action_batch

---

### 3. **Metrics to Compute**

After rolling out all 8 eval environments:

```python
metrics = {
    "eval/avg_cost": mean(agent_final_costs),
    "eval/avg_baseline_cost": mean(baseline_final_costs),
    "eval/avg_improvement_pct": mean((baseline - agent) / baseline * 100),
    "eval/avg_improvement_km": mean(baseline - agent),
    "eval/failure_rate": mean(solver_failed),
    "eval/avg_accepted_rate": mean(accepted_counts / num_customers),
}
```

**Also track per-environment**:
- Min/max improvement across 8 envs (variance)
- Best/worst episode costs

---

### 4. **When to Evaluate**

**Option A**: Every epoch (current proposal)
- Pros: Full visibility into training
- Cons: Slightly slower training

**Option B**: Every N epochs
- Pros: Faster training
- Cons: Miss some detail

**Recommendation**: Start with every epoch, can reduce to every 2-5 epochs if too slow

---

## Implementation Steps

### Step 1: Create Helper Function for Eval Envs

```python
def create_eval_envs(
    num_eval_envs: int = 8,
    num_customers: int = 30,
    eval_seed: int = 9999,
    traveler_decisions_path: Path = None,
    device: str = "cpu"
):
    """
    Create fixed evaluation environments for consistent testing.

    Returns:
        eval_agent_env: VectorizedDVRPEnv for agent rollouts
        eval_baseline_env: VectorizedDVRPEnv for baseline rollouts
    """
    eval_agent_env = VectorizedDVRPEnv(
        num_envs=num_eval_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=eval_seed,
        traveler_decisions_path=traveler_decisions_path,
        device=device,
    )

    eval_baseline_env = VectorizedDVRPEnv(
        num_envs=num_eval_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=eval_seed,  # Same seed!
        traveler_decisions_path=traveler_decisions_path,
        device=device,
    )

    return eval_agent_env, eval_baseline_env
```

---

### Step 2: Create Evaluation Function

```python
def evaluate_epoch(
    agent: PPOAgent,
    embedding_model: EmbeddingFFN,
    eval_agent_env: VectorizedDVRPEnv,
    eval_baseline_env: VectorizedDVRPEnv,
    epoch: int,
    num_customers: int = 30,
    flexibility_personalities: list = None,
    device: str = "cpu"
) -> dict:
    """
    Evaluate agent on fixed test environments (greedy rollout).

    Args:
        agent: Trained PPO agent
        embedding_model: Embedding model for predicting flexibility
        eval_agent_env: Fixed evaluation environments for agent
        eval_baseline_env: Fixed evaluation environments for baseline
        epoch: Current training epoch
        num_customers: Number of customers per episode
        flexibility_personalities: List of flexibility type names
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    num_eval_envs = eval_agent_env.num_envs

    # Reset both eval environments
    agent_states, _ = eval_agent_env.reset()
    baseline_states, _ = eval_baseline_env.reset()

    # Track statistics
    agent_episode_costs = []
    baseline_episode_costs = []
    agent_accepted_counts = np.zeros(num_eval_envs)
    agent_failures = np.zeros(num_eval_envs)

    # Greedy rollout (epsilon=0)
    for step in range(num_customers):
        # Get user IDs
        user_ids = eval_agent_env.get_current_user_ids()

        # Predict flexibility and compute masks
        with torch.no_grad():
            user_embedding_ids = torch.LongTensor(user_ids) - 1
            pred_proba = embedding_model(user_embedding_ids)
            dist = torch.distributions.Categorical(probs=pred_proba)
            predicted_flexibilities = dist.sample()
            masks = eval_agent_env.get_masks(user_ids, predicted_flexibilities)

        # Agent actions (GREEDY: epsilon=0)
        agent_actions = agent.select_action_batch(
            agent_states,
            masks=masks,
            epsilon=0.0  # ← GREEDY!
        )

        # Baseline actions (always action 12)
        baseline_actions = np.full(num_eval_envs, 12, dtype=np.int64)

        # Step environments (NO storing in agent buffer!)
        # We don't want eval data in training buffer
        baseline_next_states, _, _, _, baseline_infos = eval_baseline_env.step(baseline_actions)

        # Get baseline marginal costs for reward computation (though we don't use rewards in eval)
        baseline_current_costs = np.array([info.get('current_cost', 0.0) for info in baseline_infos])
        baseline_marginal_costs = np.zeros(num_eval_envs)  # Dummy, not needed for eval

        agent_next_states, _, _, _, agent_infos = eval_agent_env.step(agent_actions, baseline_marginal_costs)

        # Track statistics
        for i in range(num_eval_envs):
            if agent_infos[i].get('accepted', False):
                agent_accepted_counts[i] += 1

        # Update states
        agent_states = agent_next_states
        baseline_states = baseline_next_states

    # Collect final statistics
    for i in range(num_eval_envs):
        agent_cost = agent_infos[i].get('current_cost', float('inf'))
        baseline_cost = baseline_infos[i].get('current_cost', float('inf'))
        failed = agent_infos[i].get('solver_failed', False)

        agent_episode_costs.append(agent_cost if not failed else float('inf'))
        baseline_episode_costs.append(baseline_cost)
        agent_failures[i] = 1 if failed else 0

    # Compute metrics
    valid_agent_costs = [c for c in agent_episode_costs if not np.isinf(c)]
    valid_baseline_costs = [baseline_episode_costs[i] for i, c in enumerate(agent_episode_costs) if not np.isinf(c)]

    if len(valid_agent_costs) > 0:
        avg_agent_cost = np.mean(valid_agent_costs)
        avg_baseline_cost = np.mean(valid_baseline_costs)
        improvements = [(b - a) / b * 100 for a, b in zip(valid_agent_costs, valid_baseline_costs) if b > 0]
        avg_improvement_pct = np.mean(improvements) if len(improvements) > 0 else 0.0
        avg_improvement_km = avg_baseline_cost - avg_agent_cost
    else:
        avg_agent_cost = float('inf')
        avg_baseline_cost = float('inf')
        avg_improvement_pct = 0.0
        avg_improvement_km = 0.0

    failure_rate = np.mean(agent_failures)
    avg_accepted_rate = np.mean(agent_accepted_counts / num_customers)

    # Create metrics dictionary
    metrics = {
        "eval/avg_cost": avg_agent_cost,
        "eval/avg_baseline_cost": avg_baseline_cost,
        "eval/avg_improvement_pct": avg_improvement_pct,
        "eval/avg_improvement_km": avg_improvement_km,
        "eval/failure_rate": failure_rate,
        "eval/avg_accepted_rate": avg_accepted_rate,
        "eval/min_improvement_pct": min(improvements) if len(improvements) > 0 else 0.0,
        "eval/max_improvement_pct": max(improvements) if len(improvements) > 0 else 0.0,
    }

    # Print summary
    print(f"\n[Eval Epoch {epoch}] Greedy rollout on {num_eval_envs} fixed envs:")
    print(f"  Avg Cost: {avg_agent_cost:.2f} km (Baseline: {avg_baseline_cost:.2f} km)")
    print(f"  Avg Improvement: {avg_improvement_pct:.2f}% ({avg_improvement_km:.2f} km)")
    print(f"  Acceptance Rate: {avg_accepted_rate * 100:.1f}%")
    print(f"  Failure Rate: {failure_rate * 100:.1f}%")

    return metrics
```

---

### Step 3: Integrate into Training Loop

**Location**: In `train()` function, after PPO update

```python
# In meta_train.py train() function

# Create eval environments ONCE at start
print("\nCreating fixed evaluation environments...")
eval_agent_env, eval_baseline_env = create_eval_envs(
    num_eval_envs=8,
    num_customers=num_customers,
    eval_seed=9999,
    traveler_decisions_path=traveler_decisions_path,
    device=device
)

# ... training loop ...

for epoch in range(start_epoch, num_epochs + 1):
    # ... training code ...

    # Perform PPO update
    if epoch % policy_update_interval == 0:
        train_stats = agent.update(...)

    # EVALUATE AFTER EACH EPOCH
    eval_metrics = evaluate_epoch(
        agent=agent,
        embedding_model=embedding_model,
        eval_agent_env=eval_agent_env,
        eval_baseline_env=eval_baseline_env,
        epoch=epoch,
        num_customers=num_customers,
        flexibility_personalities=flexibility_personalities,
        device=device
    )

    # Log eval metrics to wandb
    wandb.log({
        "epoch": epoch,
        **eval_metrics  # Unpack eval metrics
    })

    # ... rest of training loop ...

# Clean up at end
eval_agent_env.close()
eval_baseline_env.close()
```

---

### Step 4: Important Considerations

#### A. **Don't Store Eval Data in Agent Buffer**

During evaluation, we call `agent.select_action_batch()` which stores data in the agent's buffer.

**Problem**: Eval data will contaminate training buffer!

**Solution**: Clear buffer after eval or use a separate method

**Option 1**: Clear buffer after eval
```python
# After evaluate_epoch()
agent.clear_buffer()  # Discard eval data
```

**Option 2**: Create separate eval method (better!)
```python
def select_action_batch_eval(self, states, masks=None):
    """Select actions without storing in buffer (for evaluation)."""
    # Same as select_action_batch but don't append to buffers
    # ... (duplicate code without the appending part)
```

**Recommendation**: Use Option 1 (clear buffer) - simpler and eval is infrequent

---

#### B. **Handle Embedding Model Updates**

Embedding model updates during training, so eval uses latest embeddings.

**This is correct behavior**: We want to evaluate with current embeddings.

---

#### C. **Wandb Logging Keys**

Use consistent prefixes:
- Training metrics: `avg_reward`, `avg_cost`, `avg_improvement_pct`
- Eval metrics: `eval/avg_cost`, `eval/avg_improvement_pct`

This allows easy filtering in wandb dashboard.

---

## Testing Plan

### Unit Test: Eval Function

```python
# test_eval.py
def test_evaluate_epoch():
    # Create dummy agent, embedding, envs
    # Run evaluate_epoch
    # Check metrics are computed correctly
    # Check no data leaked into agent buffer
    pass
```

### Integration Test: Full Training with Eval

```bash
# Run 2 epochs with eval
python meta_train.py --episodes 128 --num-envs 64 --log-interval 1
```

**Expected output**:
```
[Epoch 1/2]
  Avg Reward: -12.5
  Avg Cost: 450.2 km
  ...

[Eval Epoch 1] Greedy rollout on 8 fixed envs:
  Avg Cost: 445.3 km (Baseline: 480.1 km)
  Avg Improvement: 7.24% (34.8 km)
  Acceptance Rate: 65.2%
  Failure Rate: 0.0%

[Epoch 2/2]
  ...

[Eval Epoch 2] Greedy rollout on 8 fixed envs:
  Avg Cost: 440.1 km (Baseline: 480.1 km)  ← Should improve!
  Avg Improvement: 8.33% (40.0 km)
  ...
```

---

## Performance Impact

**Evaluation overhead per epoch**:
- 8 envs × 30 steps = 240 environment steps
- ~0.5-1 second per epoch (negligible!)

**Total overhead for 100 epochs**: ~1 minute (acceptable)

---

## Summary of Changes

| File | Changes |
|------|---------|
| `meta_train.py` | Add `create_eval_envs()`, `evaluate_epoch()`, integrate into training loop |
| `ppo_agent.py` | No changes needed (already has epsilon=0 option) |
| `vectorized_env.py` | No changes needed |

**Total new code**: ~150 lines (mostly in evaluate_epoch function)

---

## Next Steps

1. ✅ Implement `create_eval_envs()` helper
2. ✅ Implement `evaluate_epoch()` function
3. ✅ Integrate into training loop
4. ✅ Test with 2-epoch run
5. ✅ Verify wandb logging
6. ✅ Run full training and monitor eval metrics

Ready to implement?
