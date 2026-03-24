# PDPTW Attention Policy -- Encoding & Decoding

Reference implementation: `oracle_policy.py`, based on [Kool et al. (2019)](https://arxiv.org/abs/1803.08475).

---

## Notation

| Symbol | Description | Default |
|--------|-------------|---------|
| $B$ | Batch size | -- |
| $N$ | Number of customers | 30 |
| $2N$ | Number of nodes (pickup + dropoff per customer) | 60 |
| $n$ | Total nodes including depot ($2N + 1$) | 61 |
| $d$ | Embedding dimension | 128 |
| $M$ | Number of attention heads | 8 |
| $d_k = d/M$ | Per-head dimension | 16 |
| $L$ | Number of encoder layers | 3 |
| $d_{\mathrm{ff}}$ | Feed-forward hidden dimension | 512 |
| $T$ | Temperature | 1.0 |
| $C$ | Tanh clipping bound | 10.0 |

---

## 1. Encoding

The encoder maps raw problem features into a set of node embeddings $\mathbf{H} \in \mathbb{R}^{B \times 2N \times d}$. It runs **once** per problem instance (not per decoding step).

### 1.1 Initial Embedding (`PDPTWInitEmbedding`)

For each node $i \in \{0, 1, \dots, 2N\}$ (depot + pickup/dropoff nodes), the raw feature vector is built from normalised coordinates and time windows:

**Coordinate normalisation** (instance-wise min-max to $[0,1]$):

$$
\hat{x}_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}, \qquad
\hat{y}_i = \frac{y_i - y_{\min}}{y_{\max} - y_{\min}}
$$

where $(x_{\min}, x_{\max})$ and $(y_{\min}, y_{\max})$ are the coordinate extremes across all nodes in the instance.

**Time-window normalisation** (instance-wise min-max to $[0,1]$):

$$
\hat{e}_i = \frac{e_i - \tau_{\min}}{\tau_{\max} - \tau_{\min}}, \qquad
\hat{l}_i = \frac{l_i - \tau_{\min}}{\tau_{\max} - \tau_{\min}}
$$

where $\tau_{\min}$ and $\tau_{\max}$ are the global minimum and maximum over all time-window values in the instance.

**Feature vector:**

$$
\mathbf{x}_i = \Big[\;\underbrace{\hat{x}_i,\; \hat{y}_i}_{\text{norm coords}}
\;;\;\underbrace{\hat{e}_i,\; \hat{l}_i}_{\text{norm TW}}
\;;\;\underbrace{d_i}_{\text{demand}}\;\Big]
\;\in\;\mathbb{R}^{5}
$$

where $d_i \in \{+1, 0, -1\}$ is the demand ($+1$ pickup, $-1$ dropoff, $0$ depot).

A single linear projection maps this to the embedding space:

$$
\mathbf{h}_i^{(0)} = W_{\text{init}}\,\mathbf{x}_i + \mathbf{b}_{\text{init}}, \qquad W_{\text{init}} \in \mathbb{R}^{d \times 5}
$$

Stacked over all nodes: $\mathbf{H}^{(0)} \in \mathbb{R}^{B \times n \times d}$, where $n = 2N + 1$.

### 1.2 Transformer Encoder (`GraphAttentionNetwork`)

The encoder is a stack of $L$ identical layers. Each layer $l = 1, \dots, L$ consists of a multi-head self-attention (MHA) sub-layer followed by a position-wise feed-forward network (FFN), both wrapped with residual connections and batch normalization.

#### Multi-Head Self-Attention

Queries, keys, and values are computed jointly from a single linear projection:

$$
[\,\mathbf{Q}^{(l)}\;;\;\mathbf{K}^{(l)}\;;\;\mathbf{V}^{(l)}\,]
\;=\;
\mathbf{H}^{(l-1)}\,W_{qkv}^{(l)},
\qquad
W_{qkv}^{(l)} \in \mathbb{R}^{d \times 3d}
$$

These are split into $M$ heads. For head $m$:

$$
\mathbf{Q}_m^{(l)},\;\mathbf{K}_m^{(l)},\;\mathbf{V}_m^{(l)} \;\in\; \mathbb{R}^{B \times 2N \times d_k}
$$

Scaled dot-product attention per head:

$$
\text{Attn}_m^{(l)}
\;=\;
\text{softmax}\!\left(\frac{\mathbf{Q}_m^{(l)}\,(\mathbf{K}_m^{(l)})^\top}{\sqrt{d_k}}\right)\mathbf{V}_m^{(l)}
\;\in\;
\mathbb{R}^{B \times 2N \times d_k}
$$

Heads are concatenated and projected:

$$
\text{MHA}^{(l)}(\mathbf{H}^{(l-1)})
\;=\;
\big[\,\text{Attn}_1^{(l)};\;\dots\;;\;\text{Attn}_M^{(l)}\,\big]\,W_O^{(l)},
\qquad
W_O^{(l)} \in \mathbb{R}^{d \times d}
$$

#### Feed-Forward Network

$$
\text{FFN}^{(l)}(\mathbf{z})
\;=\;
W_2^{(l)}\,\text{ReLU}\!\big(W_1^{(l)}\,\mathbf{z} + \mathbf{b}_1^{(l)}\big) + \mathbf{b}_2^{(l)}
$$

where $W_1^{(l)} \in \mathbb{R}^{d \times d_{\mathrm{ff}}}$ and $W_2^{(l)} \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$.

#### Residual + Normalization

$$
\hat{\mathbf{H}}^{(l)} = \text{BN}\!\big(\mathbf{H}^{(l-1)} + \text{MHA}^{(l)}(\mathbf{H}^{(l-1)})\big)
$$

$$
\mathbf{H}^{(l)} = \text{BN}\!\big(\hat{\mathbf{H}}^{(l)} + \text{FFN}^{(l)}(\hat{\mathbf{H}}^{(l)})\big)
$$

### 1.3 Encoder Output

The final encoder output is:

$$
\mathbf{H} = \mathbf{H}^{(L)} \in \mathbb{R}^{B \times 2N \times d}
$$

---

## 2. Decoder Cache (Precomputed)

Before decoding begins, several quantities are precomputed from $\mathbf{H}$ and reused at every step:

**Decoder keys, values, and logit keys** (single linear projection, split three ways):

$$
[\,\mathbf{K}_{\text{dec}}\;;\;\mathbf{V}_{\text{dec}}\;;\;\mathbf{K}_{\text{logit}}\,]
\;=\;
\mathbf{H}\,W_{\text{proj}},
\qquad
W_{\text{proj}} \in \mathbb{R}^{d \times 3d}
$$

giving $\mathbf{K}_{\text{dec}},\;\mathbf{V}_{\text{dec}},\;\mathbf{K}_{\text{logit}} \in \mathbb{R}^{B \times 2N \times d}$.

The node embeddings $\mathbf{H}$ are also cached and reused by the context embedding at every decode step. No graph-level projection is precomputed; the graph context is computed dynamically inside the query construction (§3.1).

---

## 3. Decoding (Autoregressive)

The decoder produces one action (node selection) per step. At step $t$, the partial solution so far determines the dynamic state. Decoding continues until all instances in the batch have reached a terminal state (`done = True`).

### 3.1 Query Construction (`PDPTWContextEmbedding`)

The query is built from **six embedding-sized signals**, capturing global problem structure, current position, temporal state, and constraint information.

**Signal 1 -- Graph embedding** (mean over all node embeddings):

$$
\mathbf{g} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{h}_i \;\in\; \mathbb{R}^{B \times d}
$$

**Signal 2 -- Last-node embedding** (current vehicle position):

$$
\mathbf{h}_{c_t} = \mathbf{H}[\text{current\_node}_t] \;\in\; \mathbb{R}^{B \times d}
$$

**Signal 3 -- Depot embedding** (route anchor):

$$
\mathbf{h}_0 = \mathbf{H}[:,\,0,:] \;\in\; \mathbb{R}^{B \times d}
$$

**Signal 4 -- Time signal** (current time scaled by a learnable vector):

$$
\mathbf{t}_t = \tau_t \cdot \mathbf{w}_{\text{time}} \;\in\; \mathbb{R}^{B \times d}
$$

where $\mathbf{w}_{\text{time}} \in \mathbb{R}^{d}$ is a learnable parameter (initialised from $\mathcal{N}(0, 0.01)$).

**Signal 5 -- Masked-node embedding** (mean of currently infeasible nodes):

$$
\mathbf{m}_t = \frac{\sum_{i:\,\neg\text{mask}_{t,i}}\mathbf{h}_i}{\max\!\big(|\{i:\neg\text{mask}_{t,i}\}|,\;1\big)}
\;\in\; \mathbb{R}^{B \times d}
$$

**Signal 6 -- Pending-node embedding** (mean of picked-up-but-not-yet-delivered nodes):

$$
\mathbf{p}_t = \frac{\sum_{j \in \mathcal{P}_t} \mathbf{h}_j}{\max(|\mathcal{P}_t|,\;1)}
\;\in\; \mathbb{R}^{B \times d}
$$

where $\mathcal{P}_t$ is the set of node indices currently on the vehicle (from `pending_schedule`).

**Concatenation and projection (2-layer MLP):**

$$
\mathbf{c}_t = [\,\mathbf{g}\;;\;\mathbf{h}_{c_t}\;;\;\mathbf{h}_0\;;\;\mathbf{t}_t\;;\;\mathbf{m}_t\;;\;\mathbf{p}_t\,]
\;\in\; \mathbb{R}^{B \times 6d}
$$

$$
\mathbf{q}_t = W_2\,\text{ReLU}(W_1\,\mathbf{c}_t + \mathbf{b}_1) + \mathbf{b}_2
\;\in\; \mathbb{R}^{B \times d}
$$

where $W_1 \in \mathbb{R}^{2d \times 6d}$, $W_2 \in \mathbb{R}^{d \times 2d}$.

### 3.2 Pointer Attention (`PointerAttention`)

Pointer attention computes a **scalar compatibility score** between the query and each candidate node, producing the raw action logits.

**Step 1 -- Split into heads:**

$$
\mathbf{q}_t^{(m)},\;\mathbf{K}_{\text{dec}}^{(m)},\;\mathbf{V}_{\text{dec}}^{(m)}
\;\in\;
\mathbb{R}^{B \times \cdot \times d_k}
\qquad
\text{for } m = 1, \dots, M
$$

**Step 2 -- Glimpse attention** (multi-head attention over candidate nodes, masked by the action mask $\mathbf{m}_t$):

$$
\alpha_{t,m}
\;=\;
\text{softmax}\!\left(
\frac{\mathbf{q}_t^{(m)}\,(\mathbf{K}_{\text{dec}}^{(m)})^\top}{\sqrt{d_k}}
\;+\;\mathbf{M}_t
\right)
\;\in\;
\mathbb{R}^{B \times 1 \times 2N}
$$

where $\mathbf{M}_t$ is the additive attention mask derived from the boolean action mask ($-\infty$ for infeasible nodes).

$$
\text{glimpse}_m = \alpha_{t,m}\;\mathbf{V}_{\text{dec}}^{(m)}
\;\in\;
\mathbb{R}^{B \times 1 \times d_k}
$$

**Step 3 -- Concatenate heads and project:**

$$
\mathbf{g}_t
\;=\;
W_g\,[\,\text{glimpse}_1;\;\dots\;;\;\text{glimpse}_M\,]
\;\in\;
\mathbb{R}^{B \times 1 \times d}
$$

where $W_g \in \mathbb{R}^{d \times d}$ (bias-free).

**Step 4 -- Compute raw logits** via dot product with the logit key:

$$
u_{t,i}
\;=\;
\frac{\mathbf{g}_t \cdot \mathbf{k}_{\text{logit},i}}{\sqrt{d}}
\qquad
\text{for } i = 1, \dots, 2N
$$

yielding $\mathbf{u}_t \in \mathbb{R}^{B \times 2N}$.

### 3.3 Logit Post-Processing

Three transformations are applied sequentially:

1. **Temperature scaling** ($T \neq 1$):

$$
\mathbf{u}_t \;\leftarrow\; \mathbf{u}_t \,/\, T
$$

2. **Tanh clipping** ($C > 0$):

$$
\mathbf{u}_t \;\leftarrow\; C \cdot \tanh(\mathbf{u}_t)
$$

3. **Feasibility masking:**

$$
u_{t,i} \;\leftarrow\; -\infty \qquad \text{if } \mathbf{m}_{t,i} = 0
$$

### 3.4 Action Selection

Log-probabilities:

$$
\log \mathbf{p}_t = \log\,\text{softmax}(\mathbf{u}_t) \;\in\; \mathbb{R}^{B \times 2N}
$$

Three selection strategies:

| Strategy | Rule |
|----------|------|
| **Greedy** | $a_t = \arg\max_i\; u_{t,i}$ |
| **Sampling** | $a_t \sim \text{Categorical}\!\big(\text{softmax}(\mathbf{u}_t)\big)$ |
| **Evaluate** | $a_t$ is provided externally; only log-probabilities are computed |

### 3.5 Environment Transition

After selecting action $a_t$, the environment updates the dynamic state:

$$
\text{current\_node}_{t+1},\;\tau_{t+1},\;q_{t+1},\;\mathbf{m}_{t+1},\;\text{done}_{t+1}
\;\leftarrow\;
\text{env.step}(a_t)
$$

Instances where `done = True` are frozen (forced to select the depot, action 0) for the remainder of decoding.

---

## 4. Outputs

### Log-Likelihood

The total log-probability of the generated solution $\pi = (a_1, \dots, a_{T'})$:

$$
\log p_\theta(\pi \mid s)
\;=\;
\sum_{t=1}^{T'} \log p_\theta(a_t \mid a_{1:t-1},\, s)
$$

### Entropy

$$
\mathcal{H}(\pi)
\;=\;
-\sum_{t=1}^{T'}\sum_{i=1}^{2N} p_{t,i}\,\log p_{t,i}
$$

### Reward

Computed by `env.get_reward(td, actions)` after decoding completes. Represents total route cost (travel time + penalties for vehicle usage, waiting, undelivered requests).

---

## 5. Architecture Diagram

```
                         ENCODER (runs once)
                         ==================

  Raw features per node:
  [norm_coords(2) ; norm_tw(2) ; demand(1)]  -->  Linear  -->  h_i^(0)
       (5)                                             |
                                                       v
                                              +---------------------+
                                              |   MHA Layer  x L    |
                                              |                     |
                                              |  h -> MHA -> +h -> BN|
                                              |       -> FFN -> +h -> BN|
                                              +---------------------+
                                                       |
                                                       v
                                              H in R^{B x n x d}
                                                       |
                                    +------------------+------------------+
                                    |                                     |
                                    v                                     v
                              K_dec, V_dec, K_logit                  H (cached for
                            = H @ W_proj  (split 3)                  context embedding)


                        DECODER (runs T steps)
                        ======================

  Step t:
  +--------------------+     +---------------------+     +-------------------+
  | 6-signal Context   |     | Pointer Attention    |     | Action Selection  |
  | (PDPTWContextEmb)  |     |                      |     |                   |
  |                    |     |                      |     |                   |
  | 1. mean(H)         |     |                      |     |                   |
  | 2. h_{current}     |---->| q_t  vs  K_dec       |---->| u_t / T           |
  | 3. h_{depot}       |     | --> glimpse           |     | C * tanh(.)       |
  | 4. time * w_time   |     | --> g_t . K_logit     |     | mask infeasible   |
  | 5. mean(masked)    |     +---------------------+     | softmax --> a_t   |
  | 6. mean(pending)   |                                   +-------------------+
  |                    |                                            |
  | concat(6d) -> MLP  |                                            |
  +--------------------+                                            |
       ^                                                            |
       +------------------------------------------------------------+
                              env.step(a_t)
```

---

## 6. Parameter Counts

| Component | Parameters | Formula |
|-----------|-----------|---------|
| Init embedding | $5 \cdot d + d$ | $5 \cdot 128 + 128 = 768$ |
| Encoder layer ($\times L$) | $4d^2 + 4d + 2d \cdot d_{\mathrm{ff}} + d_{\mathrm{ff}} + d + 2d$ | $\approx 197{,}000 \times 3$ |
| Context embedding (MLP + $\mathbf{w}_{\text{time}}$) | $6d \cdot 2d + 2d + 2d \cdot d + d + d$ | $196{,}864 + 32{,}896 + 128 = 229{,}888$ |
| Decoder projection | $d \cdot 3d$ | $128 \cdot 384 = 49{,}152$ |
| Pointer output projection | $d \cdot d$ | $128^2 = 16{,}384$ |
