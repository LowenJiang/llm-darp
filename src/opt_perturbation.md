# Optimal Time-Window Perturbation for the Dial-a-Ride Problem

**Reference Implementation**: `optimal_actions.py`

---

## 1. Problem Statement

We consider a single-depot, multi-vehicle Dial-a-Ride Problem (DARP) with $n$ requests, $K$ vehicles, and a discrete set of time-window perturbation actions available per request. The operator can shift each request's pickup and delivery windows before solving the routing, but traveler acceptance depends on individual flexibility preferences. The goal is to jointly select perturbation actions **and** vehicle routes to minimize total travel time.

### 1.1 Instance Structure

Nodes follow the convention from the environment:

| Node Set | Indices | Description |
|----------|---------|-------------|
| Depot | $v_0$ | Start/end for all vehicles |
| Pickups | $v_1, v_3, v_5, \ldots, v_{2n-1}$ | Odd-indexed nodes |
| Deliveries | $v_2, v_4, v_6, \ldots, v_{2n}$ | Even-indexed nodes |

Travel times are asymmetric, resolved via an H3 spatial index lookup:

$$\tau(i, j) = T\bigl[h(v_i),\; h(v_j)\bigr]$$

where $h(v_i)$ maps node $v_i$ to its H3 cell index and $T \in \mathbb{R}^{H \times H}$ is the precomputed travel-time matrix (in minutes).

### 1.2 Action Space

Each request $r \in \{1, \ldots, n\}$ may be assigned one of 16 perturbation actions $a \in \mathcal{A}$:

| Action Index | Pickup Shift (min) | Delivery Shift (min) |
|:---:|:---:|:---:|
| 0 | $-30$ | $0$ |
| 1 | $-30$ | $+10$ |
| 2 | $-30$ | $+20$ |
| 3 | $-30$ | $+30$ |
| 4 | $-20$ | $0$ |
| 5 | $-20$ | $+10$ |
| 6 | $-20$ | $+20$ |
| 7 | $-20$ | $+30$ |
| 8 | $-10$ | $0$ |
| 9 | $-10$ | $+10$ |
| 10 | $-10$ | $+20$ |
| 11 | $-10$ | $+30$ |
| 12 | $0$ | $0$ |
| 13 | $0$ | $+10$ |
| 14 | $0$ | $+20$ |
| 15 | $0$ | $+30$ |

Action 12 (no shift) is the identity / baseline action.

A negative pickup shift moves the pickup window **earlier**; a positive delivery shift moves the delivery window **later**. Both directions widen the scheduling flexibility available to the router. After shifting, windows are clamped to $[30, 1410]$ minutes (i.e., 00:30--23:30).

### 1.3 Traveler Acceptance Model

Not every action is acceptable to every traveler. Acceptance depends on:

- **Traveler identity** (ID, demographics)
- **Trip context** (purpose, origin, destination)
- **Flexibility category**: one of four types:
  1. Flexible for both early pickup and late dropoff
  2. Flexible for early pickup only
  3. Flexible for late dropoff only
  4. Inflexible for any changes

The pre-generated file `traveler_decisions_augmented.csv` records, for each (traveler, trip, shift magnitude) tuple, whether the traveler accepts or rejects the perturbation under each flexibility category. At runtime, the code filters the 16 actions down to the **accepted subset** $\mathcal{A}_r \subseteq \mathcal{A}$ for each request $r$. Action 12 (no perturbation) is always included as a fallback.

---

## 2. Cordeau-Style DARP Formulation

Both the baseline routing solver (`_solve_routing`) and the joint solver (`solve_joint_mip`) use a tightened Cordeau-style three-index MIP formulation. This section describes the common routing model.

### 2.1 Node Re-indexing

The MIP uses a separated-depot convention to model vehicle start and end:

| MIP Index | Role | Maps to TensorDict Node |
|:---------:|------|:-----------------------:|
| $0$ | Start depot | $v_0$ |
| $1, \ldots, n$ | Pickups | $v_{2r-1}$ for request $r$ |
| $n{+}1, \ldots, 2n$ | Deliveries | $v_{2r}$ for request $r$ |
| $2n{+}1$ | End depot | $v_0$ |

### 2.2 Decision Variables

| Variable | Domain | Meaning |
|----------|--------|---------|
| $x_{ijk} \in \{0,1\}$ | $(i,j) \in \mathcal{E},\ k \in \mathcal{K}$ | Vehicle $k$ traverses arc $(i,j)$ |
| $B_{ik} \geq 0$ | $i \in \mathcal{N},\ k \in \mathcal{K}$ | Arrival time of vehicle $k$ at node $i$ |

### 2.3 Objective

Minimize total travel time (in minutes):

$$\min \sum_{(i,j) \in \mathcal{E}} \sum_{k \in \mathcal{K}} c_{ij} \cdot x_{ijk}$$

where $c_{ij} = \tau(i, j)$ is the travel time from node $i$ to node $j$.

### 2.4 Routing Constraints

**(C1) Visit once.** Every pickup is visited exactly once across all vehicles:

$$\sum_{j \in \delta^+(i)} \sum_{k \in \mathcal{K}} x_{ijk} = 1 \qquad \forall\, i \in \mathcal{P}$$

**(C2) Pairing.** The same vehicle serves pickup $i$ and its delivery $n{+}i$:

$$\sum_{j \in \delta^+(i)} x_{ijk} = \sum_{j \in \delta^+(n+i)} x_{(n+i)jk} \qquad \forall\, i \in \mathcal{P},\ k \in \mathcal{K}$$

**(C3) Depot.** Every vehicle leaves the start depot exactly once and enters the end depot exactly once:

$$\sum_{j \in \delta^+(0)} x_{0jk} = 1, \quad \sum_{i \in \delta^-(2n+1)} x_{i(2n+1)k} = 1 \qquad \forall\, k \in \mathcal{K}$$

Unused vehicles satisfy this by taking the direct arc $0 \to 2n{+}1$.

**(C4) Flow conservation.** At every customer node, in-flow equals out-flow per vehicle:

$$\sum_{j \in \delta^-(i)} x_{jik} - \sum_{j \in \delta^+(i)} x_{ijk} = 0 \qquad \forall\, i \in \mathcal{P} \cup \mathcal{D},\ k \in \mathcal{K}$$

### 2.5 Time-Window Constraints

Each node has an earliest arrival $e_i$ and latest arrival $\ell_i$:

$$e_i \leq B_{ik} \leq \ell_i \qquad \forall\, i \in \mathcal{N},\ k \in \mathcal{K}$$

### 2.6 Time-Linking Constraints (Big-M with Service Time)

If vehicle $k$ traverses arc $(i,j)$, then its arrival at $j$ must be no earlier than its arrival at $i$ plus the service time $s$ at node $i$ plus the travel time $c_{ij}$:

$$B_{jk} \geq B_{ik} + c_{ij} + s - M_{ij} (1 - x_{ijk}) \qquad \forall\, (i,j) \in \mathcal{E},\ k \in \mathcal{K}$$

where $s = 1$ minute (the `SERVICE_TIME` constant) and the per-arc Big-M is:

$$M_{ij} = \max\bigl(0,\; \ell_i + c_{ij} + s - e_j\bigr)$$

**Subtour elimination.** The service time $s > 0$ is critical for correctness. Without it ($s=0$), the standard Miller--Tucker--Zemlin (MTZ) time-linking constraints fail to prevent subtours when the travel time between a pickup-delivery pair is zero (i.e., both nodes map to the same H3 cell). A disconnected cycle $i \to j \to i$ with $c_{ij} = c_{ji} = 0$ trivially satisfies $B_j \geq B_i + 0$ and $B_i \geq B_j + 0$. With $s > 0$, the cycle requires $B_i \geq B_i + 2s$, which is infeasible, thereby eliminating all subtours regardless of arc costs.

### 2.7 Precedence Cuts

A valid inequality strengthening the LP relaxation: if vehicle $k$ serves pickup $r$, its delivery $n{+}r$ must be visited after completing service at $r$ plus traveling directly:

$$B_{(n+r)k} \geq B_{rk} + c_{r,n+r} + s - M^{\text{prec}}_r \Bigl(1 - \sum_{j \in \delta^+(r)} x_{rjk}\Bigr) \qquad \forall\, r \in \mathcal{P},\ k \in \mathcal{K}$$

with $M^{\text{prec}}_r = \max(0,\; \ell_r + c_{r,n+r} + s - e_{n+r})$.

### 2.8 Arc Elimination (Preprocessing)

Before building the model, arcs that are provably time-infeasible are removed:

$$\mathcal{E} = \bigl\{(i,j) : i \neq j,\ i \neq 2n{+}1,\ j \neq 0,\ e_i + c_{ij} + s \leq \ell_j \bigr\}$$

This reduces the number of binary variables and tightens the LP relaxation.

---

## 3. Joint MIP: Routing + Action Selection

The key contribution of the solver is the **joint formulation** (`solve_joint_mip`) that simultaneously optimizes routing decisions and perturbation action selection in a single linear MIP. Because the time-window shifts are linear in the action-selection variables, the entire model remains a mixed-integer linear program.

### 3.1 Additional Decision Variables

| Variable | Domain | Meaning |
|----------|--------|---------|
| $y_{r\ell} \in \{0,1\}$ | $r \in \mathcal{P},\ \ell \in \{1,\ldots,|\mathcal{A}_r|\}$ | Request $r$ selects its $\ell$-th accepted action |

### 3.2 Action-Selection Constraint

Exactly one action per request:

$$\sum_{\ell=1}^{|\mathcal{A}_r|} y_{r\ell} = 1 \qquad \forall\, r \in \mathcal{P}$$

### 3.3 Decision-Dependent Time Windows

Each accepted action $\ell$ for request $r$ induces a specific clamped time window at both the pickup node $r$ and the delivery node $n{+}r$. Let:

- $e^P_{r\ell},\ \ell^P_{r\ell}$: pickup earliest/latest under action $\ell$
- $e^D_{r\ell},\ \ell^D_{r\ell}$: delivery earliest/latest under action $\ell$

These are precomputed as:

$$e^P_{r\ell} = \text{clamp}\bigl(e_r^{\text{base}} + \Delta^P_\ell,\; 30,\; 1410\bigr), \quad \ell^P_{r\ell} = \text{clamp}\bigl(\ell_r^{\text{base}} + \Delta^P_\ell,\; 30,\; 1410\bigr)$$

and analogously for deliveries with $\Delta^D_\ell$.

The decision-dependent time-window constraints replace the fixed TW constraints (Section 2.5) for customer nodes:

**Pickup time windows:**

$$B_{rk} \geq \sum_{\ell} e^P_{r\ell} \cdot y_{r\ell}, \qquad B_{rk} \leq \sum_{\ell} \ell^P_{r\ell} \cdot y_{r\ell} \qquad \forall\, r \in \mathcal{P},\ k \in \mathcal{K}$$

**Delivery time windows:**

$$B_{(n+r)k} \geq \sum_{\ell} e^D_{r\ell} \cdot y_{r\ell}, \qquad B_{(n+r)k} \leq \sum_{\ell} \ell^D_{r\ell} \cdot y_{r\ell} \qquad \forall\, r \in \mathcal{P},\ k \in \mathcal{K}$$

Because $\sum_\ell y_{r\ell} = 1$, the RHS of each constraint selects exactly one of the precomputed bounds, making these linear in the decision variables.

### 3.4 Widest-Window Arc Elimination

For the joint model, arc pruning must be conservative: an arc is only removed if it is infeasible under **every** possible action combination. The solver computes the widest possible time window per node across all accepted actions:

$$e_i^{\text{wide}} = \min_{\ell \in \mathcal{A}_r} e_{i\ell}, \qquad \ell_i^{\text{wide}} = \max_{\ell \in \mathcal{A}_r} \ell_{i\ell}$$

Arc elimination and Big-M values use these widest windows to ensure no feasible arc is incorrectly removed.

### 3.5 Model Size

For $n$ requests, $K$ vehicles, and average $|\mathcal{A}_r|$ accepted actions per request:

| Component | Count |
|-----------|-------|
| Arc variables $x_{ijk}$ | $|\mathcal{E}| \cdot K$ |
| Time variables $B_{ik}$ | $(2n + 2) \cdot K$ |
| Action variables $y_{r\ell}$ | $\sum_r |\mathcal{A}_r|$ |
| Total binary | $|\mathcal{E}| \cdot K + \sum_r |\mathcal{A}_r|$ |
| Total continuous | $(2n + 2) \cdot K$ |

For a typical instance with $n=30$, $K=10$, the model has approximately 30,000--40,000 binary variables.

---

## 4. Baseline Solver and Coordinate Descent

In addition to the joint MIP, two alternative solution strategies are provided.

### 4.1 Fixed-Action Baseline

The baseline fixes all requests to action 12 (no perturbation) and solves the pure routing MIP from Section 2. This yields the cost $V_0^*$ — the best achievable routing cost **without** any time-window perturbation.

### 4.2 Coordinate Descent (`solve_optimal`)

A greedy heuristic that alternates between action selection and routing:

1. Initialize all actions to 12 (no perturbation).
2. Solve the routing MIP with current actions to get cost $C$.
3. For each request $r = 1, \ldots, n$:
   - For each accepted action $a \in \mathcal{A}_r$ different from the current action:
     - Temporarily set request $r$'s action to $a$, solve routing.
     - If cost improves by at least $0.01$ min, adopt $a$ permanently.
4. Repeat until no single-request change improves cost, or max iterations reached.

This avoids the large joint MIP but may converge to local optima since it cannot simultaneously change multiple requests' actions.

---

## 5. Solution Extraction

After Gurobi reports a feasible integer solution (`SolCount > 0`):

**Action extraction.** For each request $r$, scan $y_{r\ell}$ to find the selected action index.

**Route extraction.** For each vehicle $k$, trace the route starting from the start depot (node 0) by following arcs with $x_{ijk} > 0.5$:

```
cur = 0
while cur != end_depot:
    find j such that x[cur, j, k] > 0.5
    append j to route
    cur = j
```

Gurobi node indices are mapped back to the TensorDict node convention for output.

---

## 6. Formulation Tightening Techniques

Several techniques are applied to strengthen the LP relaxation and improve solve times:

1. **Per-arc Big-M.** Rather than using a single global $M$ value, each arc $(i,j)$ uses a tailored $M_{ij} = \max(0, \ell_i + c_{ij} + s - e_j)$. Tighter Big-M values reduce the LP relaxation gap.

2. **Infeasible-arc elimination.** Arcs that violate time-window feasibility even under the most permissive conditions are removed before model construction, reducing the number of binary variables.

3. **Precedence cuts.** Valid inequalities enforcing that deliveries occur after their paired pickups plus minimum travel time. These cuts are redundant given the full formulation but tighten the LP relaxation.

4. **Service time for subtour elimination.** A 1-minute service time at each node ensures the MTZ time-linking constraints strictly increase $B$ values along any path. This prevents disconnected subtours that can arise when travel times are zero (pickup and delivery in the same H3 cell). Without service time, a cycle $i \to j \to i$ with $c_{ij} = c_{ji} = 0$ trivially satisfies $B_j \geq B_i$ and $B_i \geq B_j$; with $s=1$, the cycle requires $B_i \geq B_i + 2$, which is infeasible.

---

## 7. Usage

```bash
python src/optimal_actions.py [--customers 30] [--vehicles 10] [--time-limit 60] [--seed 42]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--customers` | 30 | Number of pickup-delivery requests $n$ |
| `--vehicles` | 10 | Maximum number of vehicles $K$ |
| `--time-limit` | 60 | Gurobi time limit in seconds |
| `--seed` | 42 | Random seed for instance generation |

The script outputs:

1. **Baseline cost** $V_0^*$: optimal routing with no perturbation (action 12 for all).
2. **Optimal cost** $V^*$: joint MIP minimizing routing cost over all feasible action-route combinations.
3. **Per-request actions**: the selected perturbation and resulting time windows.
4. **Savings**: $V_0^* - V^*$ in minutes and as a percentage.
