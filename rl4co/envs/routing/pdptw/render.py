import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib import cm, colormaps
from matplotlib.lines import Line2D

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# San Francisco area bounds (lat/lon in degrees)
SF_LAT_MIN, SF_LAT_MAX = 37.67, 37.83
SF_LON_MIN, SF_LON_MAX = -122.515, -122.35


def render(td, actions=None, ax=None):
    """
    Render PDPTW solution on San Francisco map with real GPS coordinates.

    Pickups are shown with circle markers, deliveries with triangle markers.
    Pickup-delivery pairs are connected with dashed lines.
    Routes are overlaid on SF map background.

    Coordinates expected: locs[:, 0] = latitude, locs[:, 1] = longitude
    """
    # Setup colormap for routes
    if actions is not None:
        num_routine = (actions == 0).sum().item() + 2
    else:
        num_routine = 2  # Default
    base = colormaps["nipy_spectral"]
    color_list = base(np.linspace(0, 1, num_routine))
    cmap_name = base.name + str(num_routine)
    out = base.from_list(cmap_name, color_list, num_routine)

    if ax is None:
        # Create larger figure for map
        fig, ax = plt.subplots(figsize=(14, 12))

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)
        
    # Handle action being on different device
    if actions is not None:
        actions = actions.detach().cpu()

    # if batch_size greater than 0, we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        if actions is not None:
            actions = actions[0]

    locs = td["locs"]
    demands = td["demand"]

    # Ensure demands includes depot (should be 0 at index 0)
    # demands shape should match locs shape in first dimension

    # add the depot at the first action and the end action
    if actions is not None:
        actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

    # Extract coordinates: locs[:, 0] = lat, locs[:, 1] = lon
    # For matplotlib plotting, we use (lon, lat) = (x, y)
    lats = locs[:, 0].numpy()
    lons = locs[:, 1].numpy()

    # Set axis limits to SF bounds FIRST
    ax.set_xlim(SF_LON_MIN, SF_LON_MAX)
    ax.set_ylim(SF_LAT_MIN, SF_LAT_MAX)

    # Try to add OpenStreetMap basemap (optional - can be slow)
    use_basemap = False  # Set to True to enable basemap tiles
    if use_basemap:
      try:
        import contextily as ctx
        log.info("contextily imported successfully")

        # Method 1: Try using Web Mercator projection (EPSG:3857) which is more stable
        # We need to convert our lat/lon coordinates to Web Mercator first
        try:
            from pyproj import Transformer

            # Create transformer from WGS84 (lat/lon) to Web Mercator
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

            # Transform axis limits to Web Mercator
            west_merc, south_merc = transformer.transform(SF_LON_MIN, SF_LAT_MIN)
            east_merc, north_merc = transformer.transform(SF_LON_MAX, SF_LAT_MAX)

            # Transform all location data to Web Mercator
            lons_merc, lats_merc = transformer.transform(lons, lats)

            # Update plot with Web Mercator coordinates
            ax.set_xlim(west_merc, east_merc)
            ax.set_ylim(south_merc, north_merc)

            # Update coordinate arrays for later plotting
            lons, lats = lons_merc, lats_merc
            depot_lon, depot_lat = lons[0], lats[0]
            customer_lons, customer_lats = lons[1:], lats[1:]

            log.info("Attempting to download basemap tiles...")
            ctx.add_basemap(
                ax,
                crs="EPSG:3857",
                source=ctx.providers.OpenStreetMap.Mapnik,
                attribution="© OpenStreetMap contributors",
                attribution_size=8,
                alpha=0.6,
                zorder=0
            )
            log.info("SUCCESS: Loaded OpenStreetMap basemap for San Francisco")

        except Exception as e:
            log.warning(f"Web Mercator method failed: {type(e).__name__}: {e}")
            log.warning("Trying direct tile download method...")

            # Method 2: Download tiles directly and return as image array
            try:
                # bounds2img returns (image_array, extent)
                img, ext = ctx.bounds2img(
                    SF_LON_MIN, SF_LAT_MIN, SF_LON_MAX, SF_LAT_MAX,
                    ll=True,  # lat/lon coordinates
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom='auto'
                )

                # Display the image
                ax.imshow(img, extent=ext, aspect='auto', zorder=0, alpha=0.6, interpolation='bilinear')
                log.info("SUCCESS: Loaded basemap using direct download method")
            except Exception as e2:
                log.warning(f"Direct download method also failed: {type(e2).__name__}: {e2}")
                # Will fall through to outer exception handler

      except ImportError as e:
          log.warning(f"contextily not available: {e}. Install with: pip install contextily")
          # Fallback: light blue for SF bay area water
          ax.set_facecolor('#a8d5f2')
          ax.grid(True, alpha=0.2, color='white', linewidth=0.5, zorder=0)
      except Exception as e:
          log.warning(f"Could not load basemap: {type(e).__name__}: {e}")
          log.warning("Using plain background instead.")
          ax.set_facecolor('#f0f0f0')
          ax.grid(True, alpha=0.3, zorder=0)
    else:
        # No basemap - use simple colored background
        ax.set_facecolor('#a8d5f2')  # Light blue for water
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5, zorder=0)

    # Get depot and customer locations (may have been transformed to Web Mercator above)
    depot_lon, depot_lat = lons[0], lats[0]
    customer_lons, customer_lats = lons[1:], lats[1:]

    # Separate pickups and deliveries
    # Indices: pickup 0,2,4,... delivery 1,3,5,...
    num_customers = len(customer_lons)
    num_pairs = num_customers // 2

    pickup_indices = list(range(0, num_customers, 2))
    delivery_indices = list(range(1, num_customers, 2))

    # Calculate offset for labels (proportional to map scale)
    lat_offset = (SF_LAT_MAX - SF_LAT_MIN) * 0.004  # ~0.4% of map height
    lon_offset = (SF_LON_MAX - SF_LON_MIN) * 0.005  # ~0.5% of map width

    # Plot depot
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
        label="Depot"
    )

    # Depot label
    ax.text(
        depot_lon,
        depot_lat - lat_offset,
        "DEPOT",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
        color=cm.Set2(2),
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7, edgecolor=cm.Set2(2)),
        zorder=6
    )

    # Plot pickup-delivery pairs
    for pair_idx in range(num_pairs):
        pickup_idx = pickup_indices[pair_idx]
        delivery_idx = delivery_indices[pair_idx]

        pickup_lon, pickup_lat = customer_lons[pickup_idx], customer_lats[pickup_idx]
        delivery_lon, delivery_lat = customer_lons[delivery_idx], customer_lats[delivery_idx]

        # Color for this pair
        pair_color = cm.Set1(pair_idx % 8)

        # Draw dashed line connecting pickup to delivery
        ax.plot(
            [pickup_lon, delivery_lon],
            [pickup_lat, delivery_lat],
            color=pair_color,
            linestyle='--',
            linewidth=2.5,
            alpha=0.5,
            zorder=2
        )

        # Plot pickup node (white circle)
        ax.scatter(
            pickup_lon,
            pickup_lat,
            edgecolors=pair_color,
            facecolors="white",
            s=220,
            linewidths=3.5,
            marker="o",
            alpha=1,
            zorder=4
        )

        # Plot delivery node (filled triangle)
        ax.scatter(
            delivery_lon,
            delivery_lat,
            edgecolors=pair_color,
            facecolors=pair_color,
            s=220,
            linewidths=3.5,
            marker="^",
            alpha=0.9,
            zorder=4
        )

        # Add labels with demand info
        # demands indices: 0=depot, 1=pickup0, 2=delivery0, 3=pickup1, 4=delivery1, ...
        # customer_lons indices: 0=pickup0, 1=delivery0, 2=pickup1, 3=delivery1, ...
        # So demand index = customer index + 1
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
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
            zorder=6
        )

        ax.text(
            delivery_lon,
            delivery_lat + lat_offset,
            delivery_label,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8,
            color=pair_color,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor=pair_color, alpha=0.3),
            zorder=6
        )

    # Plot route if actions are provided
    if actions is not None:
        color_idx = 0
        for action_idx in range(len(actions) - 1):
            if actions[action_idx] == 0:
                color_idx += 1

            from_lon, from_lat = lons[actions[action_idx]], lats[actions[action_idx]]
            to_lon, to_lat = lons[actions[action_idx + 1]], lats[actions[action_idx + 1]]

            # Draw route edge
            ax.plot(
                [from_lon, to_lon],
                [from_lat, to_lat],
                color=out(color_idx),
                lw=3,
                alpha=0.8,
                zorder=3
            )

            # Draw arrow
            ax.annotate(
                "",
                xy=(to_lon, to_lat),
                xytext=(from_lon, from_lat),
                arrowprops=dict(arrowstyle="-|>", color=out(color_idx), lw=3, alpha=0.8),
                size=20,
                annotation_clip=False,
                zorder=3
            )

    # Set axis properties
    ax.set_xlabel("Longitude (degrees)", fontsize=13, weight='bold')
    ax.set_ylabel("Latitude (degrees)", fontsize=13, weight='bold')
    ax.set_title(
        "PDPTW Solution - San Francisco Area\n(⚪=Pickup, ▲=Delivery, ⋯=Pair Connection)",
        fontsize=15,
        weight='bold',
        pad=15
    )
    ax.set_aspect('equal', adjustable='box')

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow',
               markeredgecolor=cm.Set2(2), markersize=14, label='Depot',
               linewidth=0, markeredgewidth=3),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=12, label='Pickup',
               linewidth=0, markeredgewidth=2.5),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
               markeredgecolor='black', markersize=12, label='Delivery',
               linewidth=0, markeredgewidth=2.5),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2,
               label='Pickup-Delivery Pair', alpha=0.6),
        Line2D([0], [0], color='blue', linewidth=3,
               label='Route', alpha=0.7)
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=11,
        framealpha=0.9,
        edgecolor='black'
    )

    # Add coordinates info
    ax.text(
        0.02, 0.02,
        f"SF Area: {SF_LAT_MIN}°-{SF_LAT_MAX}°N, {-SF_LON_MIN}°-{-SF_LON_MAX}°W",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        zorder=10
    )

    plt.tight_layout()
    return ax
