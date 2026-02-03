import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colormaps
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)

# San Francisco area bounds (lat/lon in degrees)
SF_LAT_MIN, SF_LAT_MAX = 37.67, 37.83
SF_LON_MIN, SF_LON_MAX = -122.515, -122.35


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

    if actions is not None:
        num_routine = int((actions == 0).sum().item()) + 2
    else:
        num_routine = 2
    base = colormaps["nipy_spectral"]
    color_list = base(np.linspace(0, 1, num_routine))
    cmap_name = f"{base.name}{num_routine}"
    out = base.from_list(cmap_name, color_list, num_routine)

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

    lats = locs[:, 0].numpy()
    lons = locs[:, 1].numpy()

    ax.set_xlim(SF_LON_MIN, SF_LON_MAX)
    ax.set_ylim(SF_LAT_MIN, SF_LAT_MAX)

    use_basemap = True
    if use_basemap:
        try:
            import contextily as ctx

            try:
                from pyproj import Transformer

                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                west_merc, south_merc = transformer.transform(SF_LON_MIN, SF_LAT_MIN)
                east_merc, north_merc = transformer.transform(SF_LON_MAX, SF_LAT_MAX)

                lons_merc, lats_merc = transformer.transform(lons, lats)
                ax.set_xlim(west_merc, east_merc)
                ax.set_ylim(south_merc, north_merc)
                lons, lats = lons_merc, lats_merc

                log.info("Attempting to download basemap tiles...")
                ctx.add_basemap(
                    ax,
                    crs="EPSG:3857",
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    attribution="(c) OpenStreetMap contributors",
                    attribution_size=8,
                    alpha=0.6,
                    zorder=0,
                )
                log.info("Loaded OpenStreetMap basemap for San Francisco")
            except Exception as exc:
                log.warning("Web Mercator basemap failed: %s", exc)
                try:
                    img, ext = ctx.bounds2img(
                        SF_LON_MIN,
                        SF_LAT_MIN,
                        SF_LON_MAX,
                        SF_LAT_MAX,
                        ll=True,
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom="auto",
                    )
                    ax.imshow(img, extent=ext, aspect="auto", zorder=0, alpha=0.6, interpolation="bilinear")
                    log.info("Loaded basemap using direct download")
                except Exception as exc2:
                    log.warning("Direct basemap download failed: %s", exc2)
        except ImportError as exc:
            log.warning("contextily not available: %s. Install with: pip install contextily", exc)
            ax.set_facecolor("#a8d5f2")
            ax.grid(True, alpha=0.2, color="white", linewidth=0.5, zorder=0)
        except Exception as exc:
            log.warning("Could not load basemap: %s", exc)
            ax.set_facecolor("#f0f0f0")
            ax.grid(True, alpha=0.3, zorder=0)
    else:
        ax.set_facecolor("#a8d5f2")
        ax.grid(True, alpha=0.2, color="white", linewidth=0.5, zorder=0)

    depot_lon, depot_lat = lons[0], lats[0]
    customer_lons, customer_lats = lons[1:], lats[1:]

    num_customers = len(customer_lons)
    num_pairs = num_customers // 2
    pickup_indices = list(range(0, num_customers, 2))
    delivery_indices = list(range(1, num_customers, 2))

    lat_offset = (SF_LAT_MAX - SF_LAT_MIN) * 0.004
    lon_offset = (SF_LON_MAX - SF_LON_MIN) * 0.005

    ax.scatter(
        depot_lon,
        depot_lat,
        edgecolors=cm.Set2(2),
        facecolors="yellow",
        s=350,
        linewidths=4,
        marker="s",
        alpha=0.95,
        zorder=5,
        label="Depot",
    )

    ax.text(
        depot_lon,
        depot_lat - lat_offset,
        "DEPOT",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
        color=cm.Set2(2),
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7, edgecolor=cm.Set2(2)),
        zorder=6,
    )

    for pair_idx in range(num_pairs):
        pickup_idx = pickup_indices[pair_idx]
        delivery_idx = delivery_indices[pair_idx]

        pickup_lon, pickup_lat = customer_lons[pickup_idx], customer_lats[pickup_idx]
        delivery_lon, delivery_lat = customer_lons[delivery_idx], customer_lats[delivery_idx]

        pair_color = cm.Set1(pair_idx % 8)

        ax.plot(
            [pickup_lon, delivery_lon],
            [pickup_lat, delivery_lat],
            color=pair_color,
            linestyle="--",
            linewidth=2.5,
            alpha=0.5,
            zorder=2,
        )

        ax.scatter(
            pickup_lon,
            pickup_lat,
            edgecolors=pair_color,
            facecolors="white",
            s=220,
            linewidths=3.5,
            marker="o",
            alpha=1,
            zorder=4,
        )

        ax.scatter(
            delivery_lon,
            delivery_lat,
            edgecolors=pair_color,
            facecolors=pair_color,
            s=220,
            linewidths=3.5,
            marker="^",
            alpha=0.9,
            zorder=4,
        )

        pickup_demand_idx = pickup_idx + 1
        delivery_demand_idx = delivery_idx + 1
        pickup_label = f"P{pair_idx}\n+{demands[pickup_demand_idx].item():.0f}"
        delivery_label = f"D{pair_idx}\n{demands[delivery_demand_idx].item():.0f}"

        ax.text(
            pickup_lon,
            pickup_lat + lat_offset,
            pickup_label,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8,
            color=pair_color,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            zorder=6,
        )

        ax.text(
            delivery_lon,
            delivery_lat + lat_offset,
            delivery_label,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8,
            color=pair_color,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=pair_color, alpha=0.3),
            zorder=6,
        )

    if actions is not None:
        color_idx = 0
        for action_idx in range(len(actions) - 1):
            if actions[action_idx] == 0:
                color_idx += 1

            from_lon, from_lat = lons[actions[action_idx]], lats[actions[action_idx]]
            to_lon, to_lat = lons[actions[action_idx + 1]], lats[actions[action_idx + 1]]

            ax.plot(
                [from_lon, to_lon],
                [from_lat, to_lat],
                color=out(color_idx),
                lw=3,
                alpha=0.8,
                zorder=3,
            )

            ax.annotate(
                "",
                xy=(to_lon, to_lat),
                xytext=(from_lon, from_lat),
                arrowprops=dict(arrowstyle="-|>", color=out(color_idx), lw=3, alpha=0.8),
                size=20,
                annotation_clip=False,
                zorder=3,
            )

    ax.set_xlabel("Longitude (degrees)", fontsize=13, weight="bold")
    ax.set_ylabel("Latitude (degrees)", fontsize=13, weight="bold")
    ax.set_title(
        "PDPTW Solution - San Francisco Area (Pickup=o, Delivery=^, Pair=--)",
        fontsize=15,
        weight="bold",
        pad=15,
    )
    ax.set_aspect("equal", adjustable="box")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="yellow",
            markeredgecolor=cm.Set2(2),
            markersize=14,
            label="Depot",
            linewidth=0,
            markeredgewidth=3,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=12,
            label="Pickup",
            linewidth=0,
            markeredgewidth=2.5,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=12,
            label="Delivery",
            linewidth=0,
            markeredgewidth=2.5,
        ),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Pair", alpha=0.6),
        Line2D([0], [0], color="blue", linewidth=3, label="Route", alpha=0.7),
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
