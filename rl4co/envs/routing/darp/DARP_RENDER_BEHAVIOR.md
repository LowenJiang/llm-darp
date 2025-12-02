# DARP Render Behavior

## Overview

The `render()` function in `render.py` visualizes DARP solutions by plotting node locations, vehicle routes, and pickup-dropoff pairings on a 2D coordinate space.

## Function Signature

```python
def render(td: TensorDict, actions=None, ax=None) -> matplotlib.axes.Axes
```

**Parameters:**
- `td`: TensorDict containing environment state (locations, demands, etc.)
- `actions`: Sequence of actions taken (node indices visited in order)
- `ax`: Optional matplotlib axis to plot on (creates new figure if None)

**Returns:** Matplotlib axis object with the rendered plot

## Visual Elements

### 1. Nodes (lines 144-184)

| Node Type | Marker | Color | Label Format | Description |
|-----------|--------|-------|--------------|-------------|
| **Depot** | Square (hollow) | Green | "D" | Starting/ending point for all vehicles |
| **Pickup** | Circle (hollow) | Red | "P1", "P2", ... | Pickup locations for passengers |
| **Dropoff** | Square (hollow) | Blue | "D1", "D2", ... | Dropoff locations for passengers |

All markers are hollow (unfilled) with colored edges.

### 2. Routes (lines 85-130)

**Vehicle Routes:**
- Each vehicle tour uses a **unique color** from colormap
- Solid colored arrows show the path taken
- Arrow direction indicates travel sequence
- ≤10 tours: uses `tab10` colormap
- \>10 tours: uses `nipy_spectral` colormap (lines 58-63)

**Return to Depot:**
- Grey dashed lines (`--`) show return trips to depot (lines 121-130)
- Lower opacity (alpha=0.4) to de-emphasize

### 3. Pickup-Dropoff Pairing (lines 132-142)

- **Black dashed lines** connect each pickup to its corresponding dropoff
- Very light (alpha=0.25) to avoid cluttering the main routes
- Shows the logical pairing relationship, not actual travel path

## Node Indexing & Location Extraction (lines 40-47)

The function handles the interleaved node structure:

```python
num_loc = td["locs"].shape[-2] - 1  # excluding depot
num_pickups = num_loc // 2

depot_loc = td["locs"][0]                    # Index 0: Depot
pickup_locs = td["locs"][1:num_pickups+1]    # Indices 1 to num_pickups: Pickups
dropoff_locs = td["locs"][num_pickups+1:]    # Remaining indices: Dropoffs
```

**Note:** This assumes **sequential indexing** (depot, then all pickups, then all dropoffs), which is different from the interleaved indexing in the environment logic. The generator must create `td["locs"]` in this sequential format.

## Tour Segmentation (lines 65-83)

The function splits the action sequence into separate vehicle tours:

```python
tours = []
current_tour = []

for action in actions:
    if action == 0:  # Return to depot
        if len(current_tour) > 0:
            tours.append(current_tour)
            current_tour = []
    else:  # Customer node
        current_tour.append(action.item())
```

**Logic:**
1. Depot returns (action=0) mark tour boundaries
2. Each tour gets its own color
3. Incomplete tours (no depot return) are still rendered

## Rendering Process

### Step 1: Batch Handling (lines 28-32)
If input has batch dimension, extract first batch element:
```python
if td.batch_size != torch.Size([]):
    td = td[0]
    actions = actions[0]
```

### Step 2: Color Assignment (lines 52-63)
- Count depot returns to determine number of tours
- Assign unique color per tour from matplotlib colormap

### Step 3: Plot Routes (lines 85-130)
For each tour:
1. Draw arrow from depot to first customer
2. Draw arrows between consecutive customers
3. Draw dashed grey line back to depot

### Step 4: Plot Pairing Lines (lines 132-142)
Connect each pickup to its dropoff with black dashed lines

### Step 5: Plot Nodes (lines 144-184)
- Depot (green square)
- Pickups (red circles)
- Dropoffs (blue squares)

### Step 6: Add Annotations (lines 186-223)
- "D" at depot
- "P1", "P2", ... at pickups
- "D1", "D2", ... at dropoffs

### Step 7: Formatting (lines 225-230)
- Add legend, axis labels, title
- Enable grid with low opacity
- Set equal aspect ratio

## Key Features

1. **Multi-vehicle Support**: Automatically handles multiple vehicles with distinct colors
2. **Incomplete Solutions**: Renders partial tours even if vehicle hasn't returned to depot
3. **Pairing Visualization**: Dashed lines show which pickup corresponds to which dropoff
4. **Customizable Axis**: Can render into existing matplotlib axis for subplots
5. **Automatic Scaling**: Works with any number of tours (adjusts colormap)

## Usage Example

```python
# After running environment
td_final = env.reset(batch_size=[1])
policy = YourPolicy()
out = policy(td_final)
actions = out["actions"]

# Render solution
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
env.render(td_final, actions, ax=ax)
plt.show()
```

## Limitations & Assumptions

1. **Sequential node ordering**: Assumes `td["locs"]` has depot first, then pickups, then dropoffs (not interleaved)
2. **Single batch rendering**: Only renders first instance if batch_size > 1
3. **No time information**: Doesn't visualize time windows or arrival times
4. **No capacity info**: Doesn't show vehicle loads or capacity violations
5. **Static visualization**: Doesn't show animation of route construction

## Visual Interpretation

**Good Solution Indicators:**
- All pickups and dropoffs visited (none skipped)
- Each color forms a cohesive route (not scattered)
- Pickup-dropoff pairs served by same color (same vehicle)
- Routes don't excessively backtrack

**Problem Indicators:**
- Unvisited nodes (no route arrows touching them)
- Very long routes suggesting infeasibility
- Only one color when multiple agents available (agent switching didn't occur)
