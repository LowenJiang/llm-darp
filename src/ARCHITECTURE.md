# DARP GNN Solver Architecture

**Authors**: Implementation based on POMO, GREAT, and Attention Model papers
**Date**: February 2026
**Model**: Graph Neural Network with Edge-Based Attention for Dial-a-Ride Problem

---

## Table of Contents

1. [Overview](#overview)
2. [Encoding Logic](#encoding-logic)
3. [Decoding Logic](#decoding-logic)
4. [POMO-REINFORCE Training](#pomo-reinforce-training)
5. [Implementation Details](#implementation-details)

---

## Overview

This architecture solves the Pickup and Delivery Problem with Time Windows (PDPTW) for Dial-a-Ride applications using:

- **GREAT-style encoder**: Edge-based graph attention for asymmetric travel costs
- **Attention decoder**: Context-aware pointer network with masking
- **POMO training**: Policy Optimization with Multiple Optima for variance reduction

### Problem Formulation

Given:
- Depot node: $v_0$
- Pickup nodes: $\mathcal{P} = \{v_1, v_3, v_5, \ldots\}$ (odd indices)
- Delivery nodes: $\mathcal{D} = \{v_2, v_4, v_6, \ldots\}$ (even indices)
- Pickup $v_{2i-1}$ pairs with delivery $v_{2i}$
- Travel time matrix: $T \in \mathbb{R}^{H \times H}$ (indexed by H3 spatial cells)
- Time windows: $[e_i, \ell_i]$ for each node $v_i$
- Vehicle capacity: $Q$

**Objective**: Find a sequence of actions (node visits) that minimizes total travel time while satisfying:
1. Precedence: pickup before delivery
2. Time windows: $e_i \leq \text{arrival}_i \leq \ell_i$
3. Capacity: at most $Q$ passengers onboard
4. Pairing: each pickup has corresponding delivery

---

## Encoding Logic

### 1. Edge Feature Construction

For each directed edge $(i, j)$ in the complete graph, construct a 15-dimensional feature vector:

$$
\mathbf{e}_{ij} = \begin{bmatrix}
\tau_{ij} \\
\mathbf{t}_i \\
\mathbf{t}_j \\
\delta_{ij}^{\text{pair}} \\
s_{ij}^{\text{tw}} \\
d_i \\
d_j \\
e_i / T_{\max} \\
\ell_i / T_{\max} \\
e_j / T_{\max} \\
\ell_j / T_{\max}
\end{bmatrix} \in \mathbb{R}^{15}
$$

Where:
- $\tau_{ij} = T[h_i, h_j] / T_{\max}$ : normalized travel time (via H3 lookup)
- $\mathbf{t}_i \in \{0,1\}^3$ : node type one-hot (depot, pickup, delivery)
- $\delta_{ij}^{\text{pair}} = \mathbb{1}[j = i+1 \text{ or } j = i-1]$ : pairing indicator
- $s_{ij}^{\text{tw}} = \max(0, \ell_j - e_i - \tau_{ij}) / T_{\max}$ : time window slack
- $d_i$ : demand (+1 for pickup, -1 for delivery, 0 for depot)

**Key insight**: Edge features encode both **geometric** (travel time) and **temporal** (time windows) constraints in a unified representation.

### 2. Edge Embedding

Initial embedding projects raw features to hidden dimension:

$$
\mathbf{h}_{ij}^{(0)} = \mathbf{W}_{\text{init}} \mathbf{e}_{ij} + \mathbf{b}_{\text{init}} \in \mathbb{R}^{d_h}
$$

where $d_h = 128$ is the hidden dimension.

### 3. GREAT Asymmetric Attention Layers

For $L=5$ layers, each layer $\ell$ applies separate attention mechanisms for **incoming** and **outgoing** edges.

#### In-Attention (Target Node Perspective)

For edge $(i, j)$ targeting node $j$, attend over all source nodes $k$:

$$
\mathbf{q}_{ij}^{\text{in}} = \mathbf{W}_q^{\text{in}} \mathbf{h}_{ij}^{(\ell-1)}, \quad
\mathbf{k}_{kj}^{\text{in}} = \mathbf{W}_k^{\text{in}} \mathbf{h}_{kj}^{(\ell-1)}, \quad
\mathbf{v}_{kj}^{\text{in}} = \mathbf{W}_v^{\text{in}} \mathbf{h}_{kj}^{(\ell-1)}
$$

$$
\alpha_{kj}^{\text{in}} = \frac{\exp\left(\frac{\mathbf{q}_{ij}^{\text{in}} \cdot \mathbf{k}_{kj}^{\text{in}}}{\sqrt{d_k}}\right)}{\sum_{k'} \exp\left(\frac{\mathbf{q}_{ij}^{\text{in}} \cdot \mathbf{k}_{k'j}^{\text{in}}}{\sqrt{d_k}}\right)}
$$

$$
\mathbf{m}_j^{\text{in}} = \sum_{k} \alpha_{kj}^{\text{in}} \mathbf{v}_{kj}^{\text{in}}
$$

#### Out-Attention (Source Node Perspective)

For edge $(i, j)$ from source $i$, attend over all target nodes $k$:

$$
\mathbf{q}_{ij}^{\text{out}} = \mathbf{W}_q^{\text{out}} \mathbf{h}_{ij}^{(\ell-1)}, \quad
\mathbf{k}_{ik}^{\text{out}} = \mathbf{W}_k^{\text{out}} \mathbf{h}_{ik}^{(\ell-1)}, \quad
\mathbf{v}_{ik}^{\text{out}} = \mathbf{W}_v^{\text{out}} \mathbf{h}_{ik}^{(\ell-1)}
$$

$$
\alpha_{ik}^{\text{out}} = \text{softmax}_k\left(\frac{\mathbf{q}_{ij}^{\text{out}} \cdot \mathbf{k}_{ik}^{\text{out}}}{\sqrt{d_k}}\right)
$$

$$
\mathbf{m}_i^{\text{out}} = \sum_{k} \alpha_{ik}^{\text{out}} \mathbf{v}_{ik}^{\text{out}}
$$

#### Node Aggregation

Combine in/out messages to form node embedding:

$$
\mathbf{z}_i^{(\ell)} = \text{LayerNorm}\left(\mathbf{W}_{\text{proj}}\left[\mathbf{m}_i^{\text{in}} \| \mathbf{m}_i^{\text{out}}\right] + \mathbf{b}_{\text{proj}}\right)
$$

#### Edge Reconstruction

Reconstruct edge features from source and target nodes:

$$
\mathbf{h}_{ij}^{(\ell)} = \text{LayerNorm}\left(\mathbf{W}_{\text{edge}}\left[\mathbf{z}_i^{(\ell)} \| \mathbf{z}_j^{(\ell)}\right] + \mathbf{h}_{ij}^{(\ell-1)}\right)
$$

#### Feed-Forward Network

$$
\mathbf{h}_{ij}^{(\ell)} \leftarrow \text{LayerNorm}\left(\text{FFN}(\mathbf{h}_{ij}^{(\ell)}) + \mathbf{h}_{ij}^{(\ell)}\right)
$$

where $\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x})$ with hidden dimension 256.

### 4. Final Node Embeddings

After $L$ layers, extract final node embeddings:

$$
\mathbf{Z} = \{\mathbf{z}_1^{(L)}, \mathbf{z}_2^{(L)}, \ldots, \mathbf{z}_N^{(L)}\} \in \mathbb{R}^{N \times d_h}
$$

**Output**: $\mathbf{Z}$ contains rich node representations that encode:
- Local geometry (via edge features)
- Global structure (via multi-hop attention)
- Temporal constraints (via time window features)
- Asymmetric costs (via directed attention)

---

## Decoding Logic

### 1. Context Query Construction

At each decoding step $t$, construct a context query from multiple sources:

$$
\mathbf{q}_t = \mathbf{q}_t^{\text{last}} + \mathbf{q}_t^{\text{first}} + \mathbf{q}_t^{\text{graph}} + \mathbf{q}_t^{\text{visited}} + \mathbf{q}_t^{\text{state}}
$$

#### Last Node Query

Current position embedding:

$$
\mathbf{q}_t^{\text{last}} = \mathbf{W}_{\text{last}} \mathbf{z}_{\pi_{t-1}}
$$

where $\pi_{t-1}$ is the previously visited node.

#### First Node Query

First non-depot node of current route (reset on depot return):

$$
\mathbf{q}_t^{\text{first}} = \mathbf{W}_{\text{first}} \mathbf{z}_{\pi_1}
$$

#### Graph Mean Query

Pre-computed global context:

$$
\mathbf{q}_t^{\text{graph}} = \mathbf{W}_{\text{graph}} \bar{\mathbf{z}}, \quad \bar{\mathbf{z}} = \frac{1}{N} \sum_{i=1}^N \mathbf{z}_i
$$

#### Visited Mean Query

Dynamic context from visited nodes:

$$
\mathbf{q}_t^{\text{visited}} = \mathbf{W}_{\text{visited}} \tilde{\mathbf{z}}_t, \quad \tilde{\mathbf{z}}_t = \frac{\sum_{i \in \mathcal{V}_t} \mathbf{z}_i}{|\mathcal{V}_t|}
$$

where $\mathcal{V}_t$ is the set of visited nodes at step $t$.

**Note**: For DARP, $\mathcal{V}_t$ can **shrink** when returning to depot with unresolved pickups (env retracts infeasible visits).

#### State Query

Vehicle state features projected to embedding space:

$$
\mathbf{q}_t^{\text{state}} = \mathbf{W}_{\text{state}} \begin{bmatrix}
(Q - c_t) / Q \\
\tau_t / T_{\max} \\
t / T_{\max}
\end{bmatrix}
$$

where:
- $c_t$ : current capacity used
- $\tau_t$ : current time
- $t$ : step count

### 2. Attention Score Computation

Compute compatibility scores for all candidate nodes:

$$
u_{t,j} = \frac{\mathbf{q}_t^T \mathbf{W}_k \mathbf{z}_j}{\sqrt{d_h}} - \beta \cdot \frac{\tau_{\pi_{t-1}, j}}{T_{\max} \sqrt{2}}
$$

where:
- First term: learned attention score
- Second term: **distance bias** (encourages visiting nearby nodes)
- $\beta$ : bias weight (set to 1.0)

### 3. Score Clipping

Apply hyperbolic tangent clipping for numerical stability:

$$
\tilde{u}_{t,j} = C \cdot \tanh\left(\frac{u_{t,j}}{C}\right)
$$

with clipping parameter $C = 10$.

### 4. Masking

Apply environment-computed action mask $\mathcal{M}_t$ (handles all 6 DARP constraint categories):

$$
\tilde{u}_{t,j} \leftarrow \begin{cases}
\tilde{u}_{t,j} & \text{if } j \in \mathcal{M}_t \\
-\infty & \text{otherwise}
\end{cases}
$$

The mask $\mathcal{M}_t$ enforces:
1. No revisits (except depot)
2. Precedence (pickup before delivery)
3. Capacity constraints
4. Time window feasibility (immediate + one-step lookahead)
5. Pending deliveries (must complete onboard pickups)
6. Operational rules (e.g., no depot return with passengers)

### 5. Action Probability

Apply temperature-scaled softmax:

$$
\pi_\theta(a_t = j | s_t) = \frac{\exp(\tilde{u}_{t,j} / \mathcal{T})}{\sum_{k \in \mathcal{M}_t} \exp(\tilde{u}_{t,k} / \mathcal{T})}
$$

where $\mathcal{T}$ is temperature (default 1.0, higher for exploration).

### 6. Action Selection

**Training**: Sample from distribution
$$
a_t \sim \pi_\theta(\cdot | s_t)
$$

**Inference**: Greedy selection
$$
a_t = \arg\max_{j \in \mathcal{M}_t} \tilde{u}_{t,j}
$$

---

## POMO-REINFORCE Training

### 1. POMO: Multiple Optima Exploitation

**Key Idea**: For each problem instance $x$, generate $P$ solution trajectories starting from different nodes.

#### Starting Node Selection

For DARP with $n$ customers (pickup-delivery pairs), select $P$ pickup nodes:

$$
\mathcal{S} = \{v_1, v_3, v_5, \ldots, v_{2P-1}\}
$$

If $n < P$, cycle through pickup nodes.

#### Parallel Rollouts

For instance $x^{(b)}$ in batch, generate $P$ trajectories:

$$
\tau^{(b,1)}, \tau^{(b,2)}, \ldots, \tau^{(b,P)}
$$

where trajectory $\tau^{(b,p)} = (a_1^{(b,p)}, a_2^{(b,p)}, \ldots, a_T^{(b,p)})$ with:
- $a_1^{(b,p)} = 0$ (depot)
- $a_2^{(b,p)} = s_p \in \mathcal{S}$ (POMO starting node)
- $a_t^{(b,p)} \sim \pi_\theta(\cdot | s_t^{(b,p)})$ for $t > 2$

#### Batch Expansion

State tensor is expanded for POMO:

$$
\text{TensorDict}^{[B]} \xrightarrow{\text{batchify}} \text{TensorDict}^{[B \times P]}
$$

where each element is replicated $P$ times, then diverges based on starting node.

### 2. Shared Baseline

**Key Innovation**: Use mean reward across POMO rollouts as baseline (no separate value network needed).

For instance $b$, compute shared baseline:

$$
b^{(b)} = \frac{1}{P} \sum_{p=1}^P R(\tau^{(b,p)})
$$

where $R(\tau)$ is the negative total travel time (higher is better).

### 3. Advantage Computation

For each trajectory $(b, p)$:

$$
A^{(b,p)} = \frac{R(\tau^{(b,p)}) - b^{(b)}}{\sigma^{(b)} + \epsilon}
$$

where:
- $\sigma^{(b)} = \sqrt{\frac{1}{P} \sum_{p=1}^P (R(\tau^{(b,p)}) - b^{(b)})^2}$ : standard deviation
- $\epsilon = 10^{-8}$ : numerical stability

**Normalization** reduces gradient variance across instances with different scales.

### 4. Log-Probability Computation

For trajectory $\tau^{(b,p)}$:

$$
\log \pi_\theta(\tau^{(b,p)} | x^{(b)}) = \sum_{t=1}^T \log \pi_\theta(a_t^{(b,p)} | s_t^{(b,p)})
$$

### 5. REINFORCE Gradient

Policy gradient estimator:

$$
\nabla_\theta J(\theta) \approx \frac{1}{B \cdot P} \sum_{b=1}^B \sum_{p=1}^P A^{(b,p)} \nabla_\theta \log \pi_\theta(\tau^{(b,p)} | x^{(b)})
$$

**Note**: Advantage is **detached** from computation graph (treated as constant during backward pass).

### 6. Loss Function

Minimize negative expected reward:

$$
\mathcal{L}(\theta) = -\frac{1}{B \cdot P} \sum_{b=1}^B \sum_{p=1}^P A^{(b,p)} \log \pi_\theta(\tau^{(b,p)} | x^{(b)})
$$

### 7. Optimization

**Algorithm**: Adam optimizer with gradient clipping

```python
for epoch in range(num_epochs):
    # Sample batch
    X = sample_instances(batch_size=B)

    # POMO rollout
    for b in range(B):
        for p in range(P):
            τ[b,p], log_π[b,p] = rollout(X[b], start_node=S[p], π_θ)
            R[b,p] = compute_reward(τ[b,p], X[b])

    # Compute advantages
    baseline = R.mean(dim=1, keepdim=True)  # [B, 1]
    std = R.std(dim=1, keepdim=True) + 1e-8
    advantage = (R - baseline) / std        # [B, P]

    # REINFORCE update
    loss = -(advantage.detach() * log_π).mean()
    loss.backward()
    clip_grad_norm_(θ, max_norm=1.0)
    optimizer.step()
```

**Hyperparameters**:
- Learning rate: $\alpha = 10^{-4}$
- Weight decay: $\lambda = 10^{-6}$
- Gradient clip: $\|\nabla_\theta\|_2 \leq 1.0$
- Batch size: $B = 64$
- POMO size: $P = 20$

### 8. Variance Reduction Analysis

**Standard REINFORCE** (single rollout per instance):
$$
\text{Var}[\nabla_\theta J] \propto \frac{\sigma_R^2}{B}
$$

**POMO with shared baseline**:
$$
\text{Var}[\nabla_\theta J] \propto \frac{\sigma_R^2}{B \cdot P} + \frac{\sigma_{\text{within}}^2}{B}
$$

where $\sigma_{\text{within}}^2 < \sigma_R^2$ (variance within POMO rollouts is smaller than across instances).

**Result**: POMO achieves $\approx P \times$ variance reduction compared to standard REINFORCE.

---

## Implementation Details

### File Structure

```
src/
├── trial_encoder.py          # GREAT edge-based encoder (600K params)
│   ├── EdgeFeatureConstructor  (no params)
│   ├── DARPGREATLayer          (per-layer attention)
│   └── DARPEdgeEncoder         (full encoder stack)
│
├── trial_decoder.py          # Attention decoder (82K params)
│   ├── DecoderCache            (pre-computed embeddings)
│   └── DARPDecoder             (context query + scoring)
│
├── trial_gnn_policy.py       # Complete policy (683K params)
│   └── DARPPolicy              (encoder + decoder + rollout)
│
└── trial_pomo_reinforce.py   # Training loop
    └── POMOReinforce           (POMO + REINFORCE + validation)
```

### Tensor Dimensions

| Component | Input | Output |
|-----------|-------|--------|
| EdgeFeatureConstructor | TensorDict | $[B, N, N, 15]$ |
| EdgeEmbedding | $[B, N, N, 15]$ | $[B, N, N, d_h]$ |
| DARPGREATLayer | $[B, N, N, d_h]$ | $[B, N, N, d_h]$ or $[B, N, d_h]$ |
| DARPEdgeEncoder | TensorDict | $[B, N, d_h]$ |
| DARPDecoder | $[B \times P], [B \times P, N], \ldots$ | $[B \times P, N]$ (probs) |
| DARPPolicy (forward) | TensorDict | dict with $[B \times P]$ rewards |

where:
- $B$ : batch size (typically 64)
- $N$ : number of nodes (61 for 30 customers)
- $P$ : POMO size (typically 20)
- $d_h$ : hidden dimension (128)

### Computational Complexity

**Per instance**:
- Encoding: $O(L \cdot N^2 \cdot d_h^2)$ (dominant term: edge attention)
- Decoding: $O(T \cdot N \cdot d_h)$ (sequence length $T \approx 2N$)

**Memory**:
- Edge features: $N^2 \times d_h \times 4$ bytes $\approx$ 1.9 MB (for $N=61, d_h=128$)
- Node embeddings: $N \times d_h \times 4$ bytes $\approx$ 31 KB
- POMO expansion: $P \times$ state memory

**Training time** (Apple M-series MPS):
- ~3 seconds per batch (B=64, P=20) = ~40ms per instance-rollout
- ~2 hours per epoch (2500 batches)

### Key Design Decisions

1. **Dense vs Sparse**: Use dense $[B, N, N, d_h]$ tensors for edges (simpler, faster for small $N$)
2. **Asymmetric Attention**: Separate in/out projections handle directed travel costs
3. **Environment Masking**: Trust `PDPTWEnv._compute_action_mask()` (comprehensive lookahead)
4. **Distance Bias**: Subtract normalized travel time from scores (spatial prior)
5. **Node Retraction**: Recompute visited mean each step (env can retract infeasible visits)
6. **POMO Starting Nodes**: Only pickup nodes (respect precedence constraints)

---

## Mathematical Notation Summary

| Symbol | Description |
|--------|-------------|
| $N$ | Number of nodes (1 depot + $2n$ pickup/delivery) |
| $\mathcal{P}, \mathcal{D}$ | Pickup and delivery node sets |
| $T \in \mathbb{R}^{H \times H}$ | Travel time matrix (H3 spatial cells) |
| $e_i, \ell_i$ | Earliest/latest time for node $i$ |
| $Q$ | Vehicle capacity |
| $d_h$ | Hidden dimension (128) |
| $L$ | Number of encoder layers (5) |
| $\mathbf{z}_i$ | Node embedding for node $i$ |
| $\mathbf{h}_{ij}$ | Edge embedding for edge $(i,j)$ |
| $\pi_\theta$ | Policy (probability distribution over actions) |
| $\tau$ | Trajectory (action sequence) |
| $R(\tau)$ | Reward (negative travel time) |
| $B$ | Batch size (64) |
| $P$ | POMO size (20) |
| $A^{(b,p)}$ | Advantage for instance $b$, rollout $p$ |

---

## References

1. **POMO**: Kwon et al. "POMO: Policy Optimization with Multiple Optima for Reinforcement Learning" (NeurIPS 2020)
2. **GREAT**: Drakulic et al. "GREAT: A Graph Generative Model for Routing Problems" (2021)
3. **Attention Model**: Kool et al. "Attention, Learn to Solve Routing Problems!" (ICLR 2019)
4. **DARP**: Berbeglia et al. "Dynamic pickup and delivery problems" (European Journal of Operational Research, 2010)

---

**Implementation**: February 2026
**Framework**: PyTorch 2.x with TensorDict
**Device**: MPS (Apple Silicon) / CUDA
**License**: Research Use
