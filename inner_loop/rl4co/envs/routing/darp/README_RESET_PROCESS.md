# DARPEnv Reset Process - Documentation Index

This directory contains comprehensive documentation of how the DARP environment initializes when `env.reset(batch_size=[3])` is called.

## Documents in This Series

### 1. RESET_PROCESS_SUMMARY.md (START HERE)
**Quick reference guide** - Read this first for a high-level understanding.

Contents:
- 4-step call chain overview
- Data generation process
- State initialization phases
- TensorDict structure with all fields
- Initial action mask computation
- Key insights and parameters
- ~350 lines, easy to scan

**Best for**: Understanding the overall flow and what gets returned

---

### 2. DARP_RESET_ANALYSIS.md (DETAILED)
**Complete step-by-step analysis** - Read this for deep understanding of each component.

Contents:
- Detailed call sequence with code snippets
- Generator implementation walkthrough
- Generator sampling (depot, locations, demands, time windows)
- State initialization with line-by-line breakdown
- Action mask generation algorithm
- Complete output TensorDict structure
- Node index organization
- Initial state summary table
- Vehicle switching mechanism
- Example with three different instances

**Best for**: Understanding exactly what happens at each step

---

### 3. DARP_RESET_FLOW_DIAGRAM.md (VISUAL)
**Visual flow diagrams and examples** - Read this for intuitive understanding.

Contents:
- High-level call flow diagram
- Data generation flow diagram
- State initialization flow diagram
- Action mask computation flow diagram
- TensorDict structure at each stage
- Example batch element breakdown
- Time window generation example
- Vehicle switching scenario
- Batch processing parallelism diagram

**Best for**: Visual learners, understanding the data flow

---

### 4. STEP_AND_MASKING_LOGIC.md (EXISTING)
**Already in repository** - Reference for step() and masking details.

Contents:
- Multi-vehicle routing details
- Stepping logic
- Masking constraints
- Vehicle coordination
- Edge cases and safety mechanisms

**Best for**: Understanding what happens AFTER reset (during steps)

---

## Quick Navigation Guide

### "I want to understand..."

**...the basic flow of reset()**
→ Start with RESET_PROCESS_SUMMARY.md (sections 1-4)

**...what gets returned from reset()**
→ RESET_PROCESS_SUMMARY.md (section "Returned TensorDict Structure")

**...how data is generated**
→ DARP_RESET_ANALYSIS.md (section "Data Generation: DARPGenerator._generate")
→ Or DARP_RESET_FLOW_DIAGRAM.md (section "Data Generation Flow")

**...the complete initialization process**
→ DARP_RESET_ANALYSIS.md (full document)

**...action mask logic at reset**
→ DARP_RESET_ANALYSIS.md (section "Initial Action Mask Generation")
→ Or DARP_RESET_FLOW_DIAGRAM.md (section "Action Mask Computation")

**...visual representation**
→ DARP_RESET_FLOW_DIAGRAM.md (all diagrams)

**...how stepping works (after reset)**
→ STEP_AND_MASKING_LOGIC.md

---

## File References

### Main Code Files
```
/rl4co/envs/routing/darp/env.py
├─ Line 252-328: _reset() method
└─ Line 330-412: _get_action_mask() method

/rl4co/envs/routing/darp/generator.py
└─ Line 103-181: _generate() method

/rl4co/envs/common/base.py
└─ Line 135-143: RL4COEnvBase.reset() method
```

### Documentation Files
```
/rl4co/envs/routing/darp/
├─ README_RESET_PROCESS.md (this file)
├─ RESET_PROCESS_SUMMARY.md (quick reference)
├─ DARP_RESET_ANALYSIS.md (detailed analysis)
├─ DARP_RESET_FLOW_DIAGRAM.md (visual flows)
└─ STEP_AND_MASKING_LOGIC.md (existing)
```

---

## Call Chain Overview

```
env.reset(batch_size=[3])
    ↓
RL4COEnvBase.reset()                [base.py:135-143]
    ├─ Call generator
    │  └─ DARPGenerator._generate()  [generator.py:103-181]
    │     └─ Returns: locs, depot, demand, time_windows, capacity
    └─ Call parent reset
       └─ DARPEnv._reset()           [env.py:252-328]
          ├─ Merge depot with locations
          ├─ Initialize state (zeros/False)
          ├─ Call _get_action_mask()  [env.py:330-412]
          └─ Return complete TensorDict
```

---

## TensorDict Structure at Reset

**14 fields** returned after reset():

```
TensorDict(batch_size=[3])
├─ locs: [3, 21, 2]                  Locations (depot + customers)
├─ current_node: [3, 1]              Active vehicle's current node
├─ current_agent: [3, 1]             Active vehicle index
├─ current_time: [3, 1]              Active vehicle's current time
├─ current_load: [3, 1]              Active vehicle's load
├─ visited: [3, 21]                  Which nodes visited (global)
├─ picked_up_by_current: [3, 21]    Which pickups done by this vehicle
├─ vehicle_finished: [3, 5]          Which vehicles completed tours
├─ total_distance: [3]               Total distance so far
├─ time_windows: [3, 21]             Deadline to visit each node
├─ demand: [3, 21]                   +ve for pickup, -ve for dropoff
├─ capacity: [3, 5]                  Vehicle capacity
├─ i: [3, 1]                         Step counter
└─ action_mask: [3, 21]              Feasible actions (LAST field computed)
```

---

## Key Constants

**Default Generator Parameters:**
```python
num_loc = 20                # Customers (must be even)
num_agents = 5              # Vehicles
min_loc = 0.0, max_loc = 100.0  # Coordinate range
vehicle_capacity = 5        # Capacity per vehicle
min_demand = 1, max_demand = 3  # Pickup demand range
vehicle_speed = 25.0        # For time conversion
min_pickup_tw = 10, max_pickup_tw = 30  # Time window range
```

**Initial State (at reset):**
```python
current_node = 0            # All at depot
current_agent = 0           # Agent 0 active
current_time = 0            # All at time 0
current_load = 0            # All empty
visited = all False         # Nothing visited
picked_up_by_current = all False  # No pickups
vehicle_finished = all False      # No finished vehicles
total_distance = 0          # No distance
i = 0                       # Step counter at 0
```

**Node Structure:**
```
Index:    0      1      2      3      4      ...
Role:    DEP    P0     D0     P1     D1     ...
Pattern: (0)   (+ve)  (-ve)  (+ve)  (-ve)   ...
```

---

## Initial Action Mask Pattern

From reset(), the action_mask[0] looks like:
```
[False,  True?, False, True?, False, True?, ..., True?, False]
depot   P0    D0     P1    D1    P2        P9    D9
```

Where:
- **Depot (index 0)**: Always False on first step (requires i > 0)
- **Pickups (odd)**: True if not visited, within time window, capacity allows
- **Dropoffs (even)**: Always False initially (no pickups completed yet)

---

## Important Design Features

### Multi-Vehicle Coordination
- **visited** (global): Prevents any vehicle from revisiting a node
- **picked_up_by_current** (per-vehicle): Tracks this vehicle's pickups
- **vehicle_finished**: Records which vehicles completed tours
- Vehicles switch when returning to depot

### Time Windows
- **Pickup**: Random in [10, 30] (sampled at generation)
- **Dropoff**: Constrained by pickup_time + travel_time
- Ensures feasibility: vehicle has time for full pickup-dropoff

### Batch Efficiency
- All 3 instances process in parallel (vectorized operations)
- No loops over batch elements in most operations
- GPU-friendly tensor operations

### Interleaved Node Structure
```
Index:  0   1   2   3   4   5   6   7
Role:  DEP  P0  D0  P1  D1  P2  D2  P3
```
Simplifies pairing constraints: dropoff at index i corresponds to pickup at i-1

---

## Reading Suggestions

### For Quick Understanding (15 minutes)
1. Read this document (README_RESET_PROCESS.md)
2. Skim RESET_PROCESS_SUMMARY.md sections 1-4
3. Look at DARP_RESET_FLOW_DIAGRAM.md diagrams

### For Complete Understanding (1 hour)
1. Read RESET_PROCESS_SUMMARY.md (20 min)
2. Read DARP_RESET_ANALYSIS.md (30 min)
3. Review DARP_RESET_FLOW_DIAGRAM.md (10 min)

### For Deep Dive (2+ hours)
1. Read all above documents (1 hour)
2. Read actual code in env.py and generator.py (30 min)
3. Read STEP_AND_MASKING_LOGIC.md for context (30 min)
4. Trace through example execution mentally

---

## Common Questions

**Q: What's the shape of the returned TensorDict?**
A: batch_size=[3], with 14 fields. See "TensorDict Structure at Reset" above.

**Q: How many nodes are there?**
A: 21 total: 1 depot (index 0) + 20 customers (10 pickup-dropoff pairs)

**Q: Are all 3 batch instances identical?**
A: No. They have different:
- Depot locations
- Customer locations
- Demands
- Time windows
Same: Vehicle count (5), capacity (5)

**Q: What can the first action be?**
A: Any feasible pickup (odd indices 1,3,5,...) that satisfies:
- arrival_time ≤ time_window
- current_load + demand ≤ capacity
- Not visited
Depot not allowed on first step.

**Q: What happens after reset?**
A: Call step() with an action. See STEP_AND_MASKING_LOGIC.md for details.

**Q: How is the action_mask computed?**
A: See DARP_RESET_ANALYSIS.md section "Initial Action Mask Generation"

---

## Version Info

Created: November 2024
Updated for: RL4CO main branch
DARP Generator: Default 20 locations, 5 vehicles, capacity 5

---

## Related Documentation

In same directory:
- STEP_AND_MASKING_LOGIC.md - What happens during stepping
- DARP_DEVICE_MISMATCH_DIAGNOSIS.md - Device handling
- DARP_MASKING_LOGIC.md - Masking details
- DARP_RENDER_BEHAVIOR.md - Rendering functionality
- DARP_REWARD_LOGIC.md - Reward calculation

In parent repo:
- /rl4co/envs/common/base.py - Base environment class
- /rl4co/envs/routing/cvrptw/ - Similar VRP environment

