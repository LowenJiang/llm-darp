import logging
from typing import Optional
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)

# San Francisco area bounds (lat/lon in degrees)
SF_LAT_MIN, SF_LAT_MAX = 37.67, 37.83
SF_LON_MIN, SF_LON_MAX = -122.515, -122.35


def _setup_basemap(ax):
    """Set up SF basemap with Mercator projection.

    Returns:
        to_xy: callable (lng, lat) -> (x, y) in plot coordinates
        used_mercator: bool
    """
    ax.set_xlim(SF_LON_MIN, SF_LON_MAX)
    ax.set_ylim(SF_LAT_MIN, SF_LAT_MAX)

    try:
        import contextily as ctx

        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            west_merc, south_merc = transformer.transform(SF_LON_MIN, SF_LAT_MIN)
            east_merc, north_merc = transformer.transform(SF_LON_MAX, SF_LAT_MAX)

            ax.set_xlim(west_merc, east_merc)
            ax.set_ylim(south_merc, north_merc)

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

            # Format axis ticks as real lat/lon degrees
            from matplotlib.ticker import FuncFormatter
            inv_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

            def _lon_fmt(x, _pos):
                lon, _ = inv_transformer.transform(x, 0)
                return f"{abs(lon):.2f}\u00b0{'W' if lon < 0 else 'E'}"

            def _lat_fmt(y, _pos):
                _, lat = inv_transformer.transform(0, y)
                return f"{abs(lat):.2f}\u00b0{'N' if lat >= 0 else 'S'}"

            ax.xaxis.set_major_formatter(FuncFormatter(_lon_fmt))
            ax.yaxis.set_major_formatter(FuncFormatter(_lat_fmt))

            def to_xy(lng, lat):
                return transformer.transform(lng, lat)

            return to_xy, True

        except Exception as exc:
            log.warning("Web Mercator basemap failed: %s", exc)
            try:
                img, ext = ctx.bounds2img(
                    SF_LON_MIN, SF_LAT_MIN, SF_LON_MAX, SF_LAT_MAX,
                    ll=True, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto",
                )
                ax.imshow(img, extent=ext, aspect="auto", zorder=0, alpha=0.6, interpolation="bilinear")
                log.info("Loaded basemap using direct download")
            except Exception as exc2:
                log.warning("Direct basemap download failed: %s", exc2)

    except ImportError as exc:
        log.warning("contextily not available: %s. Install with: pip install contextily", exc)

    # Fallback: plain background
    ax.set_facecolor("#a8d5f2")
    ax.grid(True, alpha=0.2, color="white", linewidth=0.5, zorder=0)

    def to_xy(lng, lat):
        return lng, lat

    return to_xy, False


def render_h3_travel_time(
    travel_time_matrix_path: Optional[str | Path] = None,
    depot_h3: Optional[str] = None,
    ax=None,
    alpha: float = 0.5,
    cmap=None,
    title: str = "SF H3 Hexagons \u2014 Travel Time from Depot",
    show: bool = True,
):
    """Visualize H3 hexagons colored by travel time from a central depot cell.

    Args:
        travel_time_matrix_path: Path to travel_time_matrix CSV (values in seconds).
            Defaults to ``travel_time_matrix_res_7.csv`` next to this file.
        depot_h3: H3 index of the depot / reference cell.  Defaults to the
            standard SF depot (resolution-7 parent of ``89283082877ffff``).
        ax: Optional matplotlib Axes.  A new figure is created when *None*.
        alpha: Fill alpha for the hexagon patches.
        cmap: Matplotlib colormap (default ``RdYlGn_r``).
        title: Plot title.
        show: Whether to call ``plt.show()`` at the end.

    Returns:
        The matplotlib Axes used for plotting.
    """
    # --- Resolve defaults ---
    if travel_time_matrix_path is None:
        travel_time_matrix_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
    travel_time_matrix_path = Path(travel_time_matrix_path)

    if depot_h3 is None:
        depot_h3 = h3.cell_to_parent("89283082877ffff", 7)

    if cmap is None:
        cmap = cm.RdYlGn_r

    # --- Load data ---
    df = pd.read_csv(travel_time_matrix_path, index_col=0)
    h3_cells = list(df.columns)
    travel_time_minutes = df.values / 60.0

    depot_idx = h3_cells.index(depot_h3)
    times_from_depot = travel_time_minutes[depot_idx]

    # --- Set up figure with basemap ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 12))
    else:
        fig = ax.get_figure()

    to_xy, used_mercator = _setup_basemap(ax)

    # --- Build hexagon patches ---
    finite_times = times_from_depot[times_from_depot < 1e6]
    vmin = 0
    vmax = np.max(finite_times) if len(finite_times) > 0 else 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    patches = []
    colors = []
    for i, cell in enumerate(h3_cells):
        if i == depot_idx:
            continue
        boundary = h3.cell_to_boundary(cell)
        polygon_xy = [to_xy(lng, lat) for lat, lng in boundary]
        patches.append(Polygon(polygon_xy, closed=True))
        colors.append(times_from_depot[i])

    pc = PatchCollection(patches, alpha=alpha, edgecolors="black", linewidths=0.5, zorder=1)
    pc.set_array(np.array(colors))
    pc.set_cmap(cmap)
    pc.set_norm(norm)
    ax.add_collection(pc)

    # --- Depot hexagon: empty (white fill, red border) ---
    depot_boundary = h3.cell_to_boundary(depot_h3)
    depot_xy = [to_xy(lng, lat) for lat, lng in depot_boundary]
    depot_patch = Polygon(depot_xy, closed=True, facecolor="white", edgecolor="red",
                          linewidth=2.5, alpha=0.8, zorder=2)
    ax.add_patch(depot_patch)
    depot_center = np.mean(depot_xy, axis=0)
    ax.text(depot_center[0], depot_center[1], "DEPOT", ha="center", va="center",
            fontsize=9, weight="bold", color="red", zorder=3)

    # --- Colorbar legend ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Travel time from depot (minutes)", fontsize=12, weight="bold")

    ax.set_xlabel("Longitude", fontsize=13, weight="bold")
    ax.set_ylabel("Latitude", fontsize=13, weight="bold")
    ax.set_title(title, fontsize=16)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    if show:
        plt.show()
    return ax


if __name__ == "__main__":
    render_h3_travel_time()
