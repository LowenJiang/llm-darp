import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from visualize import SF_LAT_MIN, SF_LAT_MAX, SF_LON_MIN, SF_LON_MAX, _setup_basemap

log = logging.getLogger(__name__)


def _effective_actions(actions: torch.Tensor) -> torch.Tensor:
    """Drop pickups that were not delivered before returning to depot."""
    if actions.numel() == 0:
        return actions

    output: list[int] = []
    segment: list[int] = []
    pickups_seen: set[int] = set()
    delivered_pickups: set[int] = set()

    def flush_segment() -> None:
        nonlocal output, segment, pickups_seen, delivered_pickups
        if not segment:
            return
        filtered = [
            node
            for node in segment
            if not ((node % 2 != 0) and (node not in delivered_pickups))
        ]
        if filtered:
            output.extend(filtered)
            output.append(0)
        segment = []
        pickups_seen = set()
        delivered_pickups = set()

    for node in actions.tolist():
        if node == 0:
            flush_segment()
            continue
        segment.append(int(node))
        if node % 2 != 0:
            pickups_seen.add(int(node))
        else:
            partner = int(node) - 1
            if partner in pickups_seen:
                delivered_pickups.add(partner)

    flush_segment()
    while output and output[-1] == 0:
        output.pop()
    return actions.new_tensor(output, dtype=actions.dtype)


def render(td, actions: Optional[torch.Tensor] = None, ax=None):
    """
    Render PDPTW solution on San Francisco map with real GPS coordinates.

    Pickups are shown with circle markers, deliveries with triangle markers.
    Pickup-delivery pairs are connected with dashed lines.
    Routes are rendered after removing pickups that were not delivered before depot.

    Coordinates expected: locs[:, 0] = latitude, locs[:, 1] = longitude
    """
    if actions is not None and not isinstance(actions, torch.Tensor):
        actions = torch.tensor(actions)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 12))

    if hasattr(td, "detach"):
        td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)
    if actions is not None and hasattr(actions, "detach"):
        actions = actions.detach().cpu()

    if getattr(td, "batch_size", torch.Size([])) != torch.Size([]):
        td = td[0]
        if actions is not None:
            actions = actions[0]

    locs = td["locs"]
    demands = td["demand"]

    if actions is not None:
        actions = actions.to(torch.long)
        actions = _effective_actions(actions)
        actions = torch.cat(
            [torch.tensor([0], dtype=actions.dtype), actions, torch.tensor([0], dtype=actions.dtype)]
        )

    # Build route color palette after _effective_actions so depot count matches
    # Bright, saturated colors that stand out against a map basemap
    _ROUTE_COLORS = [
        "#e6194b",  # red
        "#3cb44b",  # green
        "#4363d8",  # blue
        "#f58231",  # orange
        "#911eb4",  # purple
        "#42d4f4",  # cyan
        "#f032e6",  # magenta
        "#bfef45",  # lime
        "#fabed4",  # pink
        "#469990",  # teal
        "#dcbeff",  # lavender
        "#9a6324",  # brown
        "#800000",  # maroon
        "#aaffc3",  # mint
        "#000075",  # navy
        "#ffe119",  # yellow
    ]
    if actions is not None:
        num_routes = int((actions[1:-1] == 0).sum().item()) + 1
    else:
        num_routes = 1

    def route_color(idx):
        return _ROUTE_COLORS[idx % len(_ROUTE_COLORS)]

    lats = locs[:, 0].numpy()
    lons = locs[:, 1].numpy()

    to_xy, used_mercator = _setup_basemap(ax)

    # Project node coordinates to match the basemap
    if used_mercator:
        projected = np.array([to_xy(lon, lat) for lon, lat in zip(lons, lats)])
        lons, lats = projected[:, 0], projected[:, 1]

    depot_lon, depot_lat = lons[0], lats[0]
    customer_lons, customer_lats = lons[1:], lats[1:]

    num_customers = len(customer_lons)
    num_pairs = num_customers // 2

    # --- Depot: large red-edged star ---
    ax.scatter(
        depot_lon,
        depot_lat,
        edgecolors="red",
        facecolors="yellow",
        s=600,
        linewidths=3,
        marker="*",
        alpha=1.0,
        zorder=7,
    )

    # --- Compute display positions: nudge overlapping nodes apart ---
    # Uses iterative pairwise repulsion so that ALL pairs end up non-overlapping.
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_sep = max(x_range, y_range) * 0.025

    display_lons = customer_lons.astype(np.float64).copy()
    display_lats = customer_lats.astype(np.float64).copy()

    for _iteration in range(500):
        moved = False
        for i in range(num_customers):
            for j in range(i + 1, num_customers):
                dx = display_lons[j] - display_lons[i]
                dy = display_lats[j] - display_lats[i]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < min_sep:
                    if dist < 1e-12:
                        angle = 2 * np.pi * (i % 7) / 7
                        dx, dy = np.cos(angle), np.sin(angle)
                        dist = 1e-12
                    overlap = (min_sep - dist) / 2 + 1e-9
                    ux, uy = dx / dist, dy / dist
                    display_lons[i] -= overlap * ux
                    display_lats[i] -= overlap * uy
                    display_lons[j] += overlap * ux
                    display_lats[j] += overlap * uy
                    moved = True
        if not moved:
            break

    # --- Customer nodes: uniform white circles with black edge, numbered 1..2n ---
    for node_idx in range(num_customers):
        actual_x, actual_y = customer_lons[node_idx], customer_lats[node_idx]
        disp_x, disp_y = display_lons[node_idx], display_lats[node_idx]

        # Draw connector line from display position to actual position if nudged
        if abs(disp_x - actual_x) > 1e-6 or abs(disp_y - actual_y) > 1e-6:
            ax.plot(
                [disp_x, actual_x], [disp_y, actual_y],
                color="#aaaaaa", linewidth=0.7, alpha=0.6,
                zorder=4, linestyle="-",
            )
            # Small dot at actual location
            ax.scatter(
                actual_x, actual_y,
                facecolors="#aaaaaa", edgecolors="none",
                s=15, alpha=0.6, zorder=4,
            )

        ax.scatter(
            disp_x, disp_y,
            edgecolors="black",
            facecolors="white",
            s=200,
            linewidths=1.5,
            marker="o",
            alpha=1.0,
            zorder=5,
        )
        # Node label: 1-based node index (node 0 is depot)
        ax.text(
            disp_x, disp_y,
            str(node_idx + 1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=7,
            color="black",
            weight="bold",
            zorder=6,
        )

    # --- Pickup-delivery pair dashed connectors (thin, gray) ---
    for pair_idx in range(num_pairs):
        pi = pair_idx * 2
        di = pi + 1
        ax.plot(
            [display_lons[pi], display_lons[di]],
            [display_lats[pi], display_lats[di]],
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            zorder=2,
        )

    if actions is not None:
        color_idx = 0
        for action_idx in range(len(actions) - 1):
            # Increment color on depot visits *after* the first one (route separator)
            if action_idx > 0 and actions[action_idx] == 0:
                color_idx += 1

            from_lon, from_lat = lons[actions[action_idx]], lats[actions[action_idx]]
            to_lon, to_lat = lons[actions[action_idx + 1]], lats[actions[action_idx + 1]]

            ax.annotate(
                "",
                xy=(to_lon, to_lat),
                xytext=(from_lon, from_lat),
                arrowprops=dict(arrowstyle="-|>", color=route_color(color_idx), lw=1.5, alpha=0.85),
                size=14,
                annotation_clip=False,
                zorder=3,
            )

    ax.set_xlabel("Longitude", fontsize=13, weight="bold")
    ax.set_ylabel("Latitude", fontsize=13, weight="bold")
    ax.set_title(
        "PDPTW Solution - San Francisco Area",
        fontsize=15,
        weight="bold",
        pad=15,
    )
    ax.set_aspect("equal", adjustable="box")

    legend_elements = [
        Line2D(
            [0], [0],
            marker="*", color="w",
            markerfacecolor="yellow", markeredgecolor="red",
            markersize=18, label="Depot",
            linewidth=0, markeredgewidth=2,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor="white", markeredgecolor="black",
            markersize=11, label="Node",
            linewidth=0, markeredgewidth=1.5,
        ),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Pickup-Delivery pair", alpha=0.6),
        Line2D([0], [0], color="black", linewidth=1.5, label="Route", alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11, framealpha=0.9, edgecolor="black")

    ax.text(
        0.02,
        0.02,
        f"SF Area: {SF_LAT_MIN}-{SF_LAT_MAX}N, {-SF_LON_MIN}-{(-SF_LON_MAX)}W",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        zorder=10,
    )

    plt.tight_layout()
    return ax
