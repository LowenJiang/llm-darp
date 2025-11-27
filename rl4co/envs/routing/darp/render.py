import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D

from tensordict.tensordict import TensorDict


def render(td: TensorDict, actions=None, ax=None):
    """Render DARP environment
    
    Visualizes the Dial-a-Ride Problem solution with:
    - Green square: Depot
    - Red circles: Pickup locations
    - Blue squares: Dropoff locations
    - Colored arrows: Vehicle routes (different color per vehicle)
    - Black dashed lines: Pickup-dropoff pairs
    
    Args:
        td: TensorDict with environment state
        actions: Sequence of actions taken
        ax: Matplotlib axis to plot on
    """
    markersize = 8
    
    td = td.detach().cpu()
    
    # If batch_size greater than 0, we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        if actions is not None:
            actions = actions[0]
    
    # Get actions
    actions = actions if actions is not None else td.get("action", None)
    
    if actions is None:
        raise ValueError("No actions provided for rendering")
    
    # Get number of locations (excluding depot at index 0)
    num_loc = td["locs"].shape[-2] - 1
    num_pickups = num_loc // 2

    # Depot and customer locations
    depot_loc = td["locs"][0]

    # Interleaved indexing: odd indices (1,3,5,...) are pickups, even indices (2,4,6,...) are dropoffs
    # Build explicit index tensors for clarity and correct plotting/labeling
    pickup_indices = torch.arange(1, num_loc + 1, dtype=torch.long)
    is_pickup_mask = (pickup_indices % 2 == 1)
    pickup_indices = pickup_indices[is_pickup_mask]
    dropoff_indices = pickup_indices + 1  # each dropoff immediately follows its pickup

    pickup_locs = td["locs"][pickup_indices]
    dropoff_locs = td["locs"][dropoff_indices]
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    
    # Split actions into tours (segments separated by 0 - depot visits)
    tours = []
    current_tour = []

    for action in actions:
        action_val = action.item() if torch.is_tensor(action) else action
        if action_val == 0:  # 0 is separator for different vehicle agents
            if len(current_tour) > 0:
                tours.append(current_tour)
                current_tour = []
        else:  # Customer node
            current_tour.append(action_val)
    
    # Add the last tour if it exists and hasn't returned to depot
    if len(current_tour) > 0:
        tours.append(current_tour)
    
    # Ensure we have at least one tour entry even if tours is empty
    if len(tours) == 0:
        tours = [[]]

    # Determine number of tours from parsed segments
    num_tours = len(tours)

    # Create color list for different vehicle routes (unique per tour)
    if num_tours <= 10:
        base = colormaps["tab10"]
        # Use listed discrete colors to avoid interpolation/clamping issues
        color_list = list(base.colors[:num_tours])
    else:
        base = colormaps["nipy_spectral"]
        # Sample evenly across the spectrum, avoid endpoint to reduce duplication
        color_list = base(np.linspace(0, 1, num_tours, endpoint=False))
    
    # Plot each tour with a different color and track statistics
    tour_stats = []

    for tour_idx, tour in enumerate(tours):
        if tour_idx >= len(color_list):
            color = color_list[-1]
        else:
            color = color_list[tour_idx]

        if len(tour) > 0:
            # Track nodes visited and distance
            nodes_visited = len(tour)
            total_distance = 0.0

            # Plot from depot to first location
            from_loc = depot_loc
            to_loc = td["locs"][tour[0]]
            distance = torch.sqrt(((from_loc - to_loc) ** 2).sum()).item()
            total_distance += distance

            ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]],
                   color=color, linewidth=2, alpha=0.8)
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
                annotation_clip=False,
            )

            # Plot connections between tour locations
            for i in range(len(tour) - 1):
                from_loc = td["locs"][tour[i]]
                to_loc = td["locs"][tour[i + 1]]
                distance = torch.sqrt(((from_loc - to_loc) ** 2).sum()).item()
                total_distance += distance

                ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]],
                       color=color, linewidth=2, alpha=0.8)
                ax.annotate(
                    "",
                    xy=(to_loc[0], to_loc[1]),
                    xytext=(from_loc[0], from_loc[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2),
                    annotation_clip=False,
                )

            # Plot return to depot (dashed line)
            from_loc = td["locs"][tour[-1]]
            to_loc = depot_loc
            distance = torch.sqrt(((from_loc - to_loc) ** 2).sum()).item()
            total_distance += distance

            ax.plot(
                [from_loc[0], to_loc[0]],
                [from_loc[1], to_loc[1]],
                color="grey",
                linestyle="--",
                linewidth=1.5,
                alpha=0.4,
            )

            # Store stats for this tour
            tour_stats.append({
                'color': color,
                'nodes': nodes_visited,
                'distance': total_distance
            })
    
    # Plot pickup-dropoff pair connections (light dashed lines) using interleaved pairing
    for i in range(num_pickups):
        pickup_idx = 2 * i + 1
        dropoff_idx = 2 * i + 2
        pickup_loc = td["locs"][pickup_idx]
        dropoff_loc = td["locs"][dropoff_idx]
        ax.plot(
            [pickup_loc[0], dropoff_loc[0]],
            [pickup_loc[1], dropoff_loc[1]],
            "k--",
            alpha=0.25,
            linewidth=1,
        )
    
    # Plot depot (solid yellow star)
    ax.scatter(
        [depot_loc[0]],
        [depot_loc[1]],
        color="gold",
        marker="*",
        s=(markersize * 3)**2,
        label="Depot",
        edgecolors="orange",
        linewidths=1.5,
        zorder=5,
    )
    
    # Plot pickup locations (red circles)
    ax.scatter(
        pickup_locs[:, 0],
        pickup_locs[:, 1],
        color="tab:red",
        marker="o",
        s=(markersize * 1.2)**2,
        label="Pickup",
        edgecolors="tab:red",
        linewidths=2,
        facecolors="none",
        zorder=4,
    )
    
    # Plot dropoff locations (blue squares)
    ax.scatter(
        dropoff_locs[:, 0],
        dropoff_locs[:, 1],
        color="tab:blue",
        marker="s",
        s=(markersize * 1.2)**2,
        label="Dropoff",
        edgecolors="tab:blue",
        linewidths=2,
        facecolors="none",
        zorder=4,
    )
    
    # Annotate node indices
    # Depot
    ax.annotate(
        "D",
        (depot_loc[0], depot_loc[1]),
        textcoords="offset points",
        xytext=(0, -15),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="darkorange",
    )
    
    # Pickups and Dropoffs - show pair numbers using interleaved indices
    for i in range(num_pickups):
        # Interleaved node indices for request i+1
        pickup_idx = 2 * i + 1
        dropoff_idx = 2 * i + 2

        pickup_loc = td["locs"][pickup_idx]
        dropoff_loc = td["locs"][dropoff_idx]

        # Pickup label
        ax.annotate(
            f"P{i+1}",
            (pickup_loc[0], pickup_loc[1]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=9,
            color="tab:red",
        )

        # Dropoff label
        ax.annotate(
            f"D{i+1}",
            (dropoff_loc[0], dropoff_loc[1]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=9,
            color="tab:blue",
        )
    
    # Create legend with vehicle route statistics
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
               markersize=12, markeredgecolor='orange', label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markersize=8, markeredgecolor='tab:red', markeredgewidth=2, label='Pickup'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='w',
               markersize=8, markeredgecolor='tab:blue', markeredgewidth=2, label='Dropoff'),
    ]

    # Add vehicle route information
    for idx, stats in enumerate(tour_stats):
        label = f"Vehicle {idx+1}: {stats['nodes']} nodes | {stats['distance']:.3f} dist"
        legend_elements.append(
            Line2D([0], [0], color=stats['color'], linewidth=2, label=label)
        )

    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.95, edgecolor='gray')
    ax.set_xlabel("X coordinate", fontsize=11)
    ax.set_ylabel("Y coordinate", fontsize=11)
    ax.set_title("DARP Solution", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_aspect("equal")
    
    return ax
