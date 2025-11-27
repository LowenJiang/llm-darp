# Embedding Learning Hermeneutics

This document explains how customer flexibility preferences are learned through embedding models in the DVRP-TW reinforcement learning system.

## Table of Contents
1. [Overview](#overview)
2. [Embedding Model Architecture](#embedding-model-architecture)
3. [Data Collection Pipeline](#data-collection-pipeline)
4. [Online Learning Process](#online-learning-process)
5. [Loss Function and Training](#loss-function-and-training)
6. [Integration with PPO Training](#integration-with-ppo-training)

---

## Overview

The system learns individual customer flexibility preferences through online reinforcement learning. Instead of requiring ground-truth labels, it infers customer preferences from observed (action, acceptance) pairs during PPO training.

**Key Insight**: By observing which schedule adjustments customers accept or reject, the model can infer their flexibility type (e.g., "flexible for late dropoff but inflexible for early pickup").

---

## Embedding Model Architecture

### EmbeddingFFN Class
**File**: `embedding.py:281-299`

The embedding model maps customer IDs to flexibility probability distributions:

```python
class EmbeddingFFN(nn.Module):
    def __init__(self, num_entities, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def get_embed(self):
        return self.embedding.weight

    def forward(self, entity_ids):
        emb = self.embedding(entity_ids)
        return self.ffn(emb)
```

**What it does**:
- **Input**: Customer ID (0-indexed integer)
- **Embedding layer**: Maps customer ID to a dense vector of size `embed_dim` (64)
- **Feed-forward network**: Three fully-connected layers with Tanh activations
- **Output**: Probability distribution over 4 flexibility types via Softmax

**Architecture**:
```
Customer ID (scalar)
    ↓
Embedding(30, 64)  # 30 customers, 64-dim embeddings
    ↓
Linear(64, 128) + Tanh
    ↓
Linear(128, 128) + Tanh
    ↓
Linear(128, 4) + Softmax
    ↓
Flexibility probabilities [P(type_0), P(type_1), P(type_2), P(type_3)]
```

### Flexibility Types
**File**: `embedding.py:21-26`

```python
flexibility_personalities = [
    "flexible for late dropoff, but inflexible for early pickup",      # Type 0
    "flexible for early pickup, but inflexible for late dropoff",      # Type 1
    "inflexible for any schedule changes",                             # Type 2
    "flexible for both early pickup and late dropoff"                  # Type 3
]
```

---

## Data Collection Pipeline

### Step 1: Initialize Data Buffer
**File**: `meta_train.py:194`

```python
# Fixed-size online data collection buffer (prevents memory growth)
online_data = deque(maxlen=12800)
```

**What it does**: Creates a circular buffer that automatically discards old samples when full, maintaining the most recent 12,800 observations.

### Step 2: Collect Observations During Episodes
**File**: `meta_train.py:288-299`

```python
# Collect online data from all environments and accumulate statistics
for i in range(num_envs):
    accepted = step_infos[i].get('accepted', False)
    online_data.append({
        'customer_id': user_ids[i],
        'action': actions[i],
        'accepted': accepted
    })
    # Accumulate statistics
    env_rewards[i] += rewards[i]
    if accepted:
        env_accepted_count[i] += 1
```

**What it does**:
- After each environment step, records what happened for each of the 64 parallel environments
- Stores triplet: (customer_id, action_taken, was_accepted)
- Example: `{customer_id: 15, action: 7, accepted: True}` means customer 15 accepted action 7 (early shift -20, late shift 10)

**Data flow**:
```
Environment Step
    ↓
For each of 64 parallel envs:
    Extract: user_id, action, accepted
    ↓
Append to online_data buffer
    ↓
Buffer maintains last 12,800 samples
```

---

## Online Learning Process

### Step 3: Periodic Embedding Updates
**File**: `meta_train.py:330-355`

```python
# Update embedding model every K steps
if total_steps % steps_per_embedding_update == 0 and len(online_data) > 0:
    print(f"\n[Epoch {epoch}, Step {total_steps}] Updating embedding model with {len(online_data)} samples...")

    # Convert deque to list for update
    online_data_list = list(online_data)

    # Update embedding model
    embedding_model = update_embedding_model(
        embedding_model,
        online_data_list,
        flexibility_personalities,
        ACTION_SPACE_MAP,
        num_epochs=50,
        batch_size=min(64, len(online_data_list)),
        lr=1e-3
    )

    with torch.no_grad():
        customer_ids = torch.arange(num_customers)
        pred_proba = embedding_model(customer_ids)
        predicted_flexibility = torch.argmax(pred_proba, dim=1)

    print(f"  Embedding model updated (masking enabled)")
    print(f"  Flexibility distribution: {torch.bincount(predicted_flexibility, minlength=n_flexibilities).tolist()}")
```

**What it does**:
- Every 10 epochs × 64 envs × 30 customers = 19,200 total steps
- Triggers embedding model update using accumulated observations
- Trains for 50 epochs on the buffered data
- Reports the learned flexibility distribution across all customers

### Step 4: Create Online Dataset
**File**: `embedding.py:65-190`

The `OnlineTravelerDataset` class converts raw observations into training data:

```python
class OnlineTravelerDataset(Dataset):
    def __init__(self, df_online, flexibility_personalities, action_space_map):
        df_online = df_online.copy()

        # Convert customer_id (1-indexed) to traveler_id (0-indexed)
        if 'customer_id' in df_online.columns:
            df_online['traveler_id'] = df_online['customer_id'] - 1

        self.entity_ids = torch.from_numpy(np.array(df_online["traveler_id"])).long()
        self.decisions = torch.from_numpy(np.array(df_online["accepted"].astype(int)))

        # Compute indicator matrix
        self.ind_matrix = self._compute_indicator_matrix(
            df_online["action"].values,
            df_online["accepted"].values,
            flexibility_personalities,
            action_space_map
        )
```

**What it does**: Transforms each observation into:
- `entity_ids`: Customer ID (0-indexed for embedding lookup)
- `decisions`: Binary acceptance (1 = accepted, 0 = rejected)
- `ind_matrix`: Which flexibility types are **consistent** with the observed decision

### Step 5: Compute Indicator Matrix
**File**: `embedding.py:145-165`

```python
def _compute_indicator_matrix(self, actions, accepted, flexibility_personalities, action_space_map):
    """
    Compute indicator matrix where ind_matrix[i, l] = 1 if flexibility type l
    would make the same decision as observed for sample i.
    """
    n_samples = len(actions)
    n_flex_types = len(flexibility_personalities)
    ind_matrix = torch.zeros((n_samples, n_flex_types), dtype=torch.float32)

    for i in range(n_samples):
        action_idx = actions[i]
        observed_decision = accepted[i]  # True/False or 1/0

        for l in range(n_flex_types):
            # What would this flexibility type decide?
            flex_would_accept = self._would_accept_action(action_idx, l, action_space_map)

            # Indicator is 1 if both made the same decision
            ind_matrix[i, l] = float(flex_would_accept == observed_decision)

    return ind_matrix
```

**Example**: If customer 5 **accepted** action 12 (no shift: early=0, late=0):
- Type 0 (inflexible early): would accept → indicator = 1 ✓
- Type 1 (inflexible late): would accept → indicator = 1 ✓
- Type 2 (inflexible both): would accept → indicator = 1 ✓
- Type 3 (flexible both): would accept → indicator = 1 ✓

Result: `ind_matrix[i] = [1, 1, 1, 1]` (all types consistent)

**Example 2**: If customer 7 **rejected** action 3 (early=-30, late=30):
- Type 0 (inflexible early): would reject → indicator = 1 ✓
- Type 1 (inflexible late): would reject → indicator = 1 ✓
- Type 2 (inflexible both): would reject → indicator = 1 ✓
- Type 3 (flexible both): would accept → indicator = 0 ✗

Result: `ind_matrix[i] = [1, 1, 1, 0]` (type 3 inconsistent)

### Step 6: Determine Flexibility Acceptance Rules
**File**: `embedding.py:121-143`

```python
def _would_accept_action(self, action_idx, flex_type_idx, action_space_map):
    """
    Determine if a flexibility type would accept a given action.

    Flexibility types:
        0: flexible for late dropoff, inflexible for early pickup
        1: flexible for early pickup, inflexible for late dropoff
        2: inflexible for any schedule changes
        3: flexible for both early pickup and late dropoff
    """
    early_shift, late_shift = action_space_map[action_idx]
    early_shift = abs(early_shift)  # Convert to positive value

    if flex_type_idx == 0:  # Flexible late dropoff, inflexible early pickup
        return early_shift == 0
    elif flex_type_idx == 1:  # Flexible early pickup, inflexible late dropoff
        return late_shift == 0
    elif flex_type_idx == 2:  # Inflexible for any changes
        return early_shift == 0 and late_shift == 0
    elif flex_type_idx == 3:  # Flexible for both
        return True
```

**What it does**: Encodes the decision rules for each flexibility type based on the action's time shifts.

---

## Loss Function and Training

### Step 7: Likelihood Loss with Regularization
**File**: `embedding.py:301-350`

```python
def likelihood_loss(beta_matrix: torch.Tensor,
              P_z_given_d: torch.Tensor,
              ind_matrix: torch.Tensor,
              alpha_e: float = 0.5,
              alpha: float = 0.4,
              eps: float = 1e-9,
              unbiased_var: bool = False) -> torch.Tensor:
    """
    Compute loss corresponding to likelihood (we minimize this).

    Loss = - sum_{i=1..N} sum_{l=1..L} w_tilde[i,l] * log(P_z_given_d[i,l])
           + alpha * sum_{m=1..M} ( var(beta[:,m]) / mean_var - 1 )^2
    """
    ## Compute w_tilde (sample weights)
    w_tilde = (P_z_given_d * ind_matrix) / torch.sum(P_z_given_d * ind_matrix, dim = 1).reshape((-1, 1)) * (1 - alpha_e * torch.prod(ind_matrix, dim = 1)).reshape((-1, 1))

    # Weighted log-likelihood term
    logP = torch.log(P_z_given_d.clamp(min=eps))
    weighted_loglik = torch.sum(w_tilde * logP)

    # Regularization: variance equalization across embedding dimensions
    var_dims = beta_matrix.var(dim=0, unbiased=unbiased_var)
    mean_var = var_dims.mean()

    if mean_var.item() == 0.0:
        reg_term = torch.tensor(0.0, device=beta_matrix.device, dtype=beta_matrix.dtype)
    else:
        ratio = var_dims / (mean_var + 1e-12)
        reg_term = torch.sum((ratio - 1.0) ** 2)

    # Full loss: maximize (weighted_loglik + alpha * reg_term)
    # → minimize negative
    loss = - (weighted_loglik + alpha * reg_term)

    return loss
```

**What it does**:

1. **Compute Sample Weights (w_tilde)**:
   - Higher weight to flexibility types that are both:
     - Consistent with observation (`ind_matrix[i,l] = 1`)
     - Predicted as likely by the model (`P_z_given_d[i,l]` is high)
   - Down-weight ambiguous observations where all types are consistent

2. **Weighted Log-Likelihood**:
   - Maximizes probability mass on consistent flexibility types
   - Formula: `Σ_i Σ_l w_tilde[i,l] * log(P(type_l | observation_i))`

3. **Embedding Regularization**:
   - Encourages embeddings to use all dimensions equally
   - Penalizes variance imbalance across embedding dimensions
   - Prevents mode collapse where some dimensions become unused

**Loss Components**:
```
Total Loss = -(Weighted Log-Likelihood + α × Regularization)

where:
  Weighted Log-Likelihood = Σ w_tilde[i,l] * log(P_z[i,l])
  Regularization = Σ_m (var(β[:,m]) / mean_var - 1)²

  w_tilde = normalized weight favoring consistent predictions
  α = 0.4 (regularization strength)
  α_e = 0.5 (ambiguity penalty)
```

### Step 8: Training Loop
**File**: `embedding.py:240-278`

```python
try:
    dataset = OnlineTravelerDataset(df_online, flexibility_personalities, action_space_map)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=lr)

    embedding_model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for entity_ids, _, ind_matrix, _ in dataloader:
            optimizer.zero_grad()
            pred_proba = embedding_model(entity_ids)
            beta_matrix = embedding_model.get_embed()
            loss = likelihood_loss(beta_matrix, pred_proba, ind_matrix)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    embedding_model.eval()

    # Log predictions for tracked customers
    with torch.no_grad():
        tracked_ids = torch.LongTensor([cid - 1 for cid in unique_customers[:5]])
        pred_proba = embedding_model(tracked_ids)
        predicted_types = torch.argmax(pred_proba, dim=1)
        print(f"  Sample predictions for customers {unique_customers[:5]}: {predicted_types.tolist()}")
```

**What it does**:
- Standard PyTorch training loop with Adam optimizer
- Trains for 50 epochs on the online dataset
- Each iteration:
  1. Forward pass: Get predicted flexibility probabilities
  2. Compute loss using indicator matrix
  3. Backpropagate gradients
  4. Update embedding weights
- Reports sample predictions for monitoring

---

## Integration with PPO Training

### Step 9: Use Learned Embeddings for Action Masking
**File**: `meta_train.py:267-282`

```python
# Predict flexibility and compute masks based on embeddings
with torch.no_grad():
    # user_ids are 1-indexed from env, convert to 0-indexed for embedding
    user_embedding_ids = torch.LongTensor(user_ids) - 1
    pred_proba = embedding_model(user_embedding_ids)
    predicted_flexibilities = torch.argmax(pred_proba, dim=1)

    # Get action masks based on predicted flexibility
    masks = vec_env.get_masks(user_ids, predicted_flexibilities)
    masks = torch.tensor(masks)

# Select actions for all environments in parallel with masks
actions = agent.select_action_batch(
    states,
    masks=masks,
    epsilon=epsilon
)
```

**What it does**:
- Before each action selection, query the embedding model
- Get flexibility predictions for current customers
- Generate action masks that prevent invalid actions
- PPO agent selects actions only from valid (unmasked) options

**Example**:
```
Customer 12 → Embedding Model → P(types) = [0.05, 0.1, 0.05, 0.8]
                                          ↓
                            Predicted: Type 3 (flexible both)
                                          ↓
                            Mask: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] (all allowed)
                                          ↓
                            PPO selects from all 16 actions
```

### Complete Learning Cycle

```
┌─────────────────────────────────────────────────────────┐
│                    PPO Training Loop                     │
│                                                          │
│  1. Predict flexibility → Generate masks                │
│  2. Select actions (with masks)                         │
│  3. Execute actions in environment                      │
│  4. Collect (customer_id, action, accepted)            │
│  5. Store in online_data buffer                        │
│     ↓                                                    │
│  6. Every K steps: Update embedding model               │
│     - Create OnlineTravelerDataset                      │
│     - Compute indicator matrices                        │
│     - Train with likelihood loss                        │
│     - Update embeddings                                 │
│     ↓                                                    │
│  7. Use improved embeddings → Better masks              │
│     ↓                                                    │
│  (Repeat)                                               │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. **No Ground Truth Required**
The system learns customer preferences purely from observed (action, acceptance) pairs, without needing labeled flexibility types.

### 2. **Fixed-Size Buffer**
Using `deque(maxlen=12800)` prevents memory growth and naturally implements a sliding window over recent observations.

### 3. **Indicator Matrix**
Instead of hard labels, the indicator matrix represents **all flexibility types consistent** with an observation, allowing soft learning.

### 4. **Periodic Updates**
Updating every 19,200 steps (10 epochs) balances:
- Learning from enough new data
- Not updating too frequently (computational cost)
- Letting PPO policy stabilize between embedding updates

### 5. **Epsilon-Greedy Masking**
**File**: `meta_train.py:197-198, 259`

```python
initial_epsilon = 0.2
final_epsilon = 0.0
# ...
epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (epoch - 1) / max(num_epochs - 1, 1)
```

Starts with 20% chance of ignoring masks (exploration), decays to 0% (full exploitation of learned preferences).

---

## Summary

The embedding learning process creates a **self-supervised feedback loop**:

1. **Collect**: Observe customer responses to schedule adjustments
2. **Infer**: Determine which flexibility types are consistent with observations
3. **Learn**: Update embeddings to predict flexibility types
4. **Apply**: Use predictions to mask invalid actions
5. **Improve**: Better masks → Better actions → More informative observations → Better embeddings

This approach enables the system to adapt to individual customer preferences without explicit labels, learning purely from interaction data during reinforcement learning.