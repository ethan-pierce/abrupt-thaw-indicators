"""Figure 2 — Study-area / model-training-location map (L1).

Maps the exact 19,288 locations retained in ``features_clean.csv``, rather than
the larger source database. The two panels use the same 25-km hexagon lattice
and shared logarithmic count scale, so their spatial sampling densities are
directly comparable.

The available repository has no citable statewide roads or hydrography layer.
Those contextual layers are deliberately not improvised here; if a sourced
layer is added later it should be drawn beneath the hexagons in a pale neutral.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from cmcrameri import cm as cmc
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

import figstyle

# --- repo paths ------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data import local_rasters as lr  # noqa: E402  (OBU_TIF path)

FEATURES = REPO / "data" / "features_clean.csv"
AK_OUTLINE = REPO / "archive" / "data" / "no-thermokarst-sites" / "alaska-outline.geojson"

DST_CRS = "EPSG:3338"          # Alaska Albers
DISPLAY_W = 1600               # target raster width (px) for the warped domain
CELL_KM = 25.0

OCEAN = "#ffffff"
LAND_EDGE = "#666666"
DOMAIN = "#e1e4e6"
# Each class density ramp is drawn from its own half of vik (warm = Abrupt,
# cool = Non-abrupt; figstyle.ABRUPT/NON_ABRUPT sample the same map's poles),
# stopping short of vik's near-white center so mincnt=1 hexagons stay legible
# against the pale domain fill.
ABRUPT_DENSITY = LinearSegmentedColormap.from_list(
    "abrupt_density", cmc.vik(np.linspace(0.62, 1.0, 256)))
NON_ABRUPT_DENSITY = LinearSegmentedColormap.from_list(
    "non_abrupt_density", cmc.vik(np.linspace(0.38, 0.0, 256)))


def load_points():
    """Load the deduplicated locations used by the trained model."""
    df = pd.read_csv(FEATURES, usecols=["Longitude", "Latitude", "Class"])
    df = df.dropna(subset=["Longitude", "Latitude", "Class"]).copy()
    df["Class"] = df["Class"].astype(int)
    if set(df["Class"].unique()) != {0, 1}:
        raise ValueError("features_clean.csv must contain Class values 0 (abrupt) and 1 (non-abrupt)")
    tf = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    df["x"], df["y"] = tf.transform(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())
    return df


def framing_extent(df, pad_km=90.0):
    pad = pad_km * 1000.0
    xmin, xmax = df["x"].min() - pad, df["x"].max() + pad
    ymin, ymax = df["y"].min() - pad, df["y"].max() + pad
    ak = gpd.read_file(AK_OUTLINE).to_crs(DST_CRS)
    # The source multipolygon includes ~50 outlying islands (Aleutians, Kodiak,
    # St. Lawrence, ...) that fall inside the display extent but hold essentially
    # no training points (1 of 19,288) and no informative context; keep only the
    # mainland (>95% of the state's area) so they don't read as separate,
    # unlabeled landmasses.
    mainland = ak.explode(index_parts=False).reset_index(drop=True)
    mainland = mainland.loc[[mainland.geometry.area.idxmax()]]
    return (xmin, ymin, xmax, ymax), mainland


def warp_obu(extent):
    """Windowed, decimated read of the Obu domain, warped to Alaska Albers."""
    xmin, ymin, xmax, ymax = extent
    tw = DISPLAY_W
    th = int(round(tw * (ymax - ymin) / (xmax - xmin)))
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, tw, th)
    dst = np.full((th, tw), np.nan, dtype="float32")
    with rasterio.open(lr.OBU_TIF) as ds:
        src_bounds = transform_bounds(DST_CRS, ds.crs, xmin, ymin, xmax, ymax, densify_pts=41)
        window = ds.window(*src_bounds).round_offsets().round_lengths()
        decim = max(1, int(min(window.width, window.height) // max(tw, th)))
        out_h, out_w = max(1, int(window.height // decim)), max(1, int(window.width // decim))
        src = ds.read(1, window=window, out_shape=(out_h, out_w),
                      resampling=Resampling.average, boundless=True, fill_value=ds.nodata).astype("float32")
        src[src == ds.nodata] = np.nan
        src_transform = ds.window_transform(window) * Affine.scale(window.width / out_w, window.height / out_h)
        reproject(src, dst, src_transform=src_transform, src_crs=ds.crs,
                  dst_transform=dst_transform, dst_crs=DST_CRS,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
    return dst, (xmin, xmax, ymin, ymax), dst_transform


def alaska_mask(ak, dst_shape, dst_transform, buffer_km=15.0):
    """Rasterize the Alaska mainland (lightly buffered) at the display grid.

    The Obu raster is circumpolar (EPSG:3995), so an unmasked domain layer
    fills permafrost ground in Chukotka and Yukon too, reading as unlabeled
    extra landmasses. The buffer absorbs small misalignment between the
    coastline vector and the raster's own land/no-data edge.
    """
    geom = ak.geometry.iloc[0].buffer(buffer_km * 1000.0)
    return rasterize([(geom, 1)], out_shape=dst_shape, transform=dst_transform,
                      fill=0, dtype="uint8").astype(bool)


def domain_rgba(dst, mask):
    from matplotlib.colors import to_rgb
    rgba = np.zeros((*dst.shape, 4), dtype="float32")
    rgba[..., :3] = to_rgb(DOMAIN)
    rgba[..., 3] = np.where(np.isfinite(dst) & (dst > 0) & mask, 1.0, 0.0)
    return rgba


def graticule(ax, extent):
    xmin, xmax, ymin, ymax = extent
    tf = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    lat_s, lon_s = np.linspace(45, 75, 300), np.linspace(-185, -125, 300)
    for lon in [-165, -155, -145]:
        gx, gy = tf.transform(np.full_like(lat_s, lon), lat_s)
        ax.plot(gx, gy, color=figstyle.MUTED, lw=0.3, alpha=0.25, zorder=1.5)
    for lat in [60, 65, 70]:
        gx, gy = tf.transform(lon_s, np.full_like(lon_s, lat))
        ax.plot(gx, gy, color=figstyle.MUTED, lw=0.3, alpha=0.25, zorder=1.5)


def scale_bar(ax, extent):
    xmin, xmax, ymin, ymax = extent
    length = 250_000.0
    span_x, span_y = xmax - xmin, ymax - ymin
    x0, y0 = xmax - length - 0.07 * span_x, ymin + 0.055 * span_y
    h = 0.009 * span_y
    ax.add_patch(Rectangle((x0, y0), length, h, facecolor=figstyle.INK,
                           edgecolor=figstyle.INK, lw=0.4, zorder=6))
    ax.annotate("250 km", (x0 + length / 2, y0 + h), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, color=figstyle.INK, zorder=6)


def setup_map(ax, ak, rgba, extent):
    xmin, xmax, ymin, ymax = extent
    ax.set_facecolor(OCEAN)
    ak.plot(ax=ax, facecolor="none", edgecolor=LAND_EDGE, linewidth=0.5, zorder=1)
    ax.imshow(rgba, extent=(xmin, xmax, ymin, ymax), origin="upper", interpolation="nearest",
              zorder=2, rasterized=True)
    graticule(ax, extent)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_edgecolor(figstyle.MUTED)
        spine.set_linewidth(0.45)


def main():
    figstyle.use()
    df = load_points()
    extent_box, ak = framing_extent(df)
    dst, extent, dst_transform = warp_obu(extent_box)
    xmin, xmax, ymin, ymax = extent
    mask = alaska_mask(ak, dst.shape, dst_transform)
    rgba = domain_rgba(dst, mask)

    # hexbin's scalar ``gridsize`` sets ny = nx / sqrt(3), which only yields
    # regular (non-stretched) hexagons when the extent is square. Our extent
    # isn't, so ny must additionally scale by the extent's own aspect ratio.
    nx = int(round((xmax - xmin) / (CELL_KM * 1000.0)))
    ny = int(round((ymax - ymin) / (CELL_KM * 1000.0) / np.sqrt(3)))
    gridsize = (nx, ny)
    abrupt, non = df[df["Class"] == 0], df[df["Class"] == 1]
    # ``extent`` is passed explicitly to every hexbin call. Without it,
    # Matplotlib derives a different bin lattice from each class's bounds,
    # making the two panels visually aligned but analytically incomparable.
    probe, probe_ax = plt.subplots()
    try:
        maxima = []
        for group in (abrupt, non):
            probe_hex = probe_ax.hexbin(group["x"], group["y"], gridsize=gridsize,
                                        extent=extent, mincnt=1)
            maxima.append(float(probe_hex.get_array().max()))
            probe_hex.remove()
    finally:
        plt.close(probe)
    norm = LogNorm(vmin=1, vmax=max(maxima))

    fig = figstyle.figure("full", height=4.1, subplots=False)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.955, bottom=0.105, wspace=0.035)

    panels = [
        (axes[0], abrupt, ABRUPT_DENSITY, "Abrupt", "a"),
        (axes[1], non, NON_ABRUPT_DENSITY, "Non-abrupt", "b"),
    ]
    for ax, group, cmap, heading, letter in panels:
        setup_map(ax, ak, rgba, extent)
        # A thin, low-alpha dark rim keeps mincnt=1 hexagons legible against the
        # pale domain fill (their fill color alone sits too close in luminance).
        hb = ax.hexbin(group["x"], group["y"], gridsize=gridsize, extent=extent,
                       mincnt=1, cmap=cmap, norm=norm, edgecolors=(0, 0, 0, 0.35),
                       linewidths=0.18, zorder=3, rasterized=True)
        ax.text(0.025, 0.975, f"({letter})  {heading}  (n = {len(group):,})",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="semibold",
                color=figstyle.INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.2))
        cax = inset_axes(ax, width="52%", height="2.8%", loc="lower center", borderpad=3.1)
        cbar = fig.colorbar(hb, cax=cax, orientation="horizontal")
        ticks = [tick for tick in (1, 10, 100, 1000) if tick <= norm.vmax]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{int(tick):,}" for tick in ticks])
        cbar.ax.tick_params(labelsize=5.8, length=3.2, width=0.6, pad=1.2, color=figstyle.INK)
        cbar.ax.minorticks_off()
        cbar.outline.set_edgecolor(figstyle.MUTED)
        cbar.outline.set_linewidth(0.35)
        scale_bar(ax, extent)
        figstyle.north_arrow(ax, x=0.94, y=0.80, size=0.075)
    # Both furniture lines are centered under the whole figure (not split one
    # per panel) so the two panels read as a single balanced footer.
    fig.legend(handles=[Patch(facecolor=DOMAIN, edgecolor=LAND_EDGE, linewidth=0.4,
                              label="Permafrost domain")],
               loc="lower center", bbox_to_anchor=(0.5, 0.026), frameon=False,
               fontsize=7, handlelength=1.4)
    fig.text(0.5, 0.008, f"Locations per {CELL_KM:.0f}-km cell · log scale",
             ha="center", va="center", fontsize=6.5, color=figstyle.MUTED)

    # tight=False: this figure has rasterized images (imshow + hexbin) in two
    # Axes, which the default tight-bbox PDF save mis-renders (see figstyle.save).
    # Margins are hand-tuned above via subplots_adjust instead.
    figstyle.save(fig, "02_study_area_map", tight=False)
    print(f"wrote 02_study_area_map.pdf/.png  (abrupt={len(abrupt):,}, non-abrupt={len(non):,})")


if __name__ == "__main__":
    main()
