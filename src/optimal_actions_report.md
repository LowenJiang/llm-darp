# Optimal Actions Logic Report

## Overview

`optimal_actions.py` finds the **best time-window perturbation actions** for each request in a Dial-a-Ride Problem (DARP) instance to minimize total routing cost. It leverages full knowledge of traveler flexibility preferences and solves the problem using Gurobi MIP.

## Key Concepts

- **Action Space**: 16 discrete actions, each a `(pickup_shift, dropoff_shift)` pair in minutes. Shifts range from -30 to 0 for pickup and 0 to +30 for dropoff, in 10-min increments. Action 12 = `(0, 0)` = no perturbation (always available as fallback).
- **Accepted Actions**: Not every action is feasible for every traveler. Acceptance depends on the traveler's flexibility type (e.g., "flexible for both early pickup and late dropoff") cross-referenced against a pre-computed decision table (`traveler_decisions_augmented.csv`).

## Pipeline (5 Steps)

### Step 1 — Generate DARP Instance
Uses `SFGenerator` to create a batch of requests with pickup/delivery nodes, time windows, H3-based locations, and a travel-time matrix.

### Step 2 — Build Accepted Actions per Request
For each request `r`:
1. Look up the traveler's flexibility type from instance data.
2. For each of the 16 actions, query `traveler_decisions_augmented.csv` to check if the traveler would **accept** that time-window shift.
3. Collect all accepted `(action_idx, pickup_shift, dropoff_shift)` tuples. Guarantee action 12 (no-op) is always included.

### Step 3 — Solve Baseline (No Perturbation)
Solve the DARP with all requests fixed at action 12 (no shift). This gives the baseline routing cost `V_0*` via `_solve_routing`, a Cordeau-style MIP with:
- Binary arc variables `x[i,j,k]` (vehicle k traverses arc i→j)
- Continuous visit-time variables `B[i,k]`
- Constraints: each pickup visited once, same vehicle for pickup-delivery pair, flow conservation, time-window bounds, Big-M travel-time linking, precedence (delivery after pickup)
- Objective: minimize total travel time

### Step 4 — Solve Joint MIP (Routing + Action Selection)
This is the core optimization (`solve_joint_mip`). It extends the routing MIP with **action-selection variables**:

- **New binary variables**: `y[r, a]` — for each request `r`, select exactly one accepted action `a`.
- **Decision-dependent time windows**: The pickup/delivery time-window bounds become **linear functions of `y`**, e.g.:
  ```
  B[r, k] >= Σ_a (e_pickup[r,a] * y[r,a])
  B[r, k] <= Σ_a (l_pickup[r,a] * y[r,a])
  ```
  Since each `y[r,a]` is binary and exactly one is selected per request, this linearizes cleanly.
- **Widest-window arc pruning**: Before building arcs, compute the widest possible time window per node (union over all actions). Arcs infeasible even under the most permissive windows are eliminated.
- **Per-arc Big-M tightening**: Big-M values are computed from the widest windows for a tighter LP relaxation.

The solver simultaneously optimizes which action to apply to each request **and** the vehicle routes, yielding the globally optimal cost `V*`.

### Step 5 — Report Results
Compares baseline vs. optimal:
- Per-request selected actions and resulting time windows
- Total cost savings `V_0* - V*` (absolute and percentage)
- Number of vehicles used in each solution

## Alternative Solver: Coordinate Descent (`solve_optimal`)

A fallback/baseline method that iterates over requests one at a time:
1. Start with all actions = 12 (no perturbation).
2. For each request, try every accepted action while holding others fixed; keep the action that most reduces total cost.
3. Repeat until no single-request change improves cost (convergence).

This is a greedy heuristic — it cannot capture joint action interactions the way the single joint MIP can.

## Summary

| Method | Approach | Optimality |
|---|---|---|
| Baseline | All actions = no-op, solve routing | Reference cost |
| Coordinate Descent | Greedy per-request action swap | Local optimum |
| **Joint MIP** | **Simultaneous routing + action selection** | **Global optimum** (within Gurobi time limit) |
