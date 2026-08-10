"""Figure 3 — Abrupt-thaw susceptibility map + AOA reliability panel (L5 + L4b).

EXPLORATORY VARIANT — deliberately departs from figstyle/STYLE.md's diverging
vik + binary-AOA convention (2026-07-20 critique): (a) uses a single-hue
sequential map so the wide range of negative (non-abrupt-favoring) values
doesn't collapse into one saturated color across most of the state; (b) shows
the continuous dissimilarity index (``aoa.nc``'s ``DI``), not the binary
inside/outside flag, so the reliability panel has real spatial structure
instead of a flat gray field. Colorbars are built locally in this script
(figstyle.py is shared across every figure and is not touched here).

The headline product. (a) statewide log-evidence susceptibility surface
(``data/susceptibility.nc``); (b) the Area-of-Applicability dissimilarity
index (``data/aoa.nc``, Meyer & Pebesma 2021 rank-CDF DI), with the AOA
threshold contoured on top — merged in as a panel so the product is never
shown without its reliability caveat (STYLE.md).

Both source rasters are regular EPSG:4326 grids (1 km GEE reproject, T37) with
per-cell lon/lat carried as coordinates; this script derives their affine
transform from those coordinates and warps into Alaska Albers (EPSG:3338) to
match Fig 2's basemap treatment, rather than reusing the flat, distorted
lon/lat imshow the modeling scripts (predict.py / aoa.py) draw for their own
diagnostic purposes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from affine import Affine
from cmcrameri import cm as cmc
from matplotlib.colors import Normalize
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from pyproj import Transformer

import figstyle
from fig02_study_area import AK_OUTLINE, graticule, scale_bar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from settings import DATA  # noqa: E402

SUSCEPTIBILITY_NC = DATA / "susceptibility.nc"
AOA_NC = DATA / "aoa.nc"

SRC_CRS = "EPSG:4326"
DST_CRS = "EPSG:3338"          # Alaska Albers, matches Fig 2
DISPLAY_W = 1600               # target raster width (px), matches Fig 2
PAD_KM = 40.0                  # context padding around the data's own footprint

OCEAN = "#ffffff"
LAND_EDGE = "#666666"

# Crameri Scientific Colour Maps only (house rule) — sequential single-hue map
# for log-evidence (pale = strongly non-abrupt-favoring, dark = strongly
# abrupt-favoring), deliberately NOT vik/diverging, so the whole range south of
# the Brooks Range reads as graded rather than "all equally blue".
SUSCEPTIBILITY_CMAP = cmc.lajolla_r    # reversed: pale = non-abrupt-favoring, dark = abrupt-favoring
DI_CMAP = cmc.oslo_r           # mono-hued blue: pale = reliable, dark = extrapolating


def load_grids():
    """Log-evidence + AOA DI rasters on their native regular EPSG:4326 grid."""
    le_ds = xr.open_dataset(SUSCEPTIBILITY_NC)
    aoa_ds = xr.open_dataset(AOA_NC)
    log_evidence = le_ds["log_evidence"].values
    di = aoa_ds["DI"].values
    threshold = float(aoa_ds.attrs["aoa_threshold"])
    lon, lat = le_ds["longitude"].values, le_ds["latitude"].values
    valid = np.isfinite(log_evidence) & np.isfinite(lon) & np.isfinite(lat)
    if not np.array_equal(valid, np.isfinite(di)):
        raise ValueError("susceptibility.nc and aoa.nc must share an identical valid mask "
                          "(both derive from predict.py's Obu domain mask)")
    return log_evidence, di, threshold, lon, lat, valid


def source_transform(lon, lat, valid):
    """Recover the source EPSG:4326 affine from the datacube's per-cell lon/lat.

    The datacube is a regular grid (GEE ``reproject(EPSG:4326, scale=1000)``),
    so pixel index and lon/lat are related by one shared affine everywhere;
    fit it by linear regression (robust to the irregular ROI boundary, where
    lon/lat are only defined on the subset of columns/rows that are ``valid``).
    """
    ys, xs = np.nonzero(valid)
    lon_fit = np.polyfit(xs, lon[ys, xs], 1)   # lon = dlon * col + lon0
    lat_fit = np.polyfit(ys, lat[ys, xs], 1)   # lat = dlat * row + lat0
    dlon, lon0 = lon_fit
    dlat, lat0 = lat_fit
    # Affine maps (col, row) -> the pixel's UPPER-LEFT corner; lon0/lat0 above are
    # cell-CENTRE values at index 0, so shift back by half a pixel.
    return Affine(dlon, 0.0, lon0 - dlon / 2.0, 0.0, dlat, lat0 - dlat / 2.0)


def dest_grid(lon, lat, valid, pad_km=PAD_KM, display_w=DISPLAY_W):
    """Alaska-Albers destination transform sized to the data's own footprint."""
    tf = Transformer.from_crs(SRC_CRS, DST_CRS, always_xy=True)
    x, y = tf.transform(lon[valid], lat[valid])
    pad = pad_km * 1000.0
    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad
    th = int(round(display_w * (ymax - ymin) / (xmax - xmin)))
    transform = from_bounds(xmin, ymin, xmax, ymax, display_w, th)
    return (xmin, xmax, ymin, ymax), (th, display_w), transform


def warp_to_albers(src, src_transform, dst_shape, dst_transform, resampling=Resampling.nearest):
    dst = np.full(dst_shape, np.nan, dtype="float64")
    reproject(np.ascontiguousarray(src, dtype="float64"), dst,
              src_transform=src_transform, src_crs=SRC_CRS,
              dst_transform=dst_transform, dst_crs=DST_CRS,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    return dst


def mainland_outline():
    import geopandas as gpd
    ak = gpd.read_file(AK_OUTLINE).to_crs(DST_CRS)
    mainland = ak.explode(index_parts=False).reset_index(drop=True)
    return mainland.loc[[mainland.geometry.area.idxmax()]]


def setup_map(ax, ak, extent):
    xmin, xmax, ymin, ymax = extent
    ax.set_facecolor(OCEAN)
    ak.plot(ax=ax, facecolor="none", edgecolor=LAND_EDGE, linewidth=0.5, zorder=1)
    graticule(ax, extent)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_edgecolor(figstyle.MUTED)
        spine.set_linewidth(0.45)


def xy_grid(extent, shape):
    """Pixel-center x/y coordinates matching an origin='upper' imshow of `shape`."""
    xmin, xmax, ymin, ymax = extent
    ny, nx = shape
    xs = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    ys = ymax - (np.arange(ny) + 0.5) * (ymax - ymin) / ny  # row 0 is north
    return xs, ys


def panel_a_susceptibility(fig, ax, cax, le_warp, extent):
    finite = le_warp[np.isfinite(le_warp)]
    vmin, vmax = np.percentile(finite, [1, 99])
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(le_warp, extent=extent, origin="upper", interpolation="nearest",
                   cmap=SUSCEPTIBILITY_CMAP, norm=norm, zorder=2, rasterized=True)
    # NOTE: a contour(levels=[0]) was tried here to mark the neutral boundary, but
    # the field is noisy at pixel scale in the high-susceptibility north slope (sign
    # flips between adjacent 1-km cells), so the contour degenerates into thousands
    # of tiny segments that paint the region solid black. The colorbar's "0" tick
    # carries that information instead.
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.set_ticklabels([f"{vmin:.1f}", "0", f"{vmax:.1f}"])
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6, color=figstyle.INK)
    cbar.outline.set_edgecolor(figstyle.MUTED)
    cbar.outline.set_linewidth(0.35)
    cbar.set_label("Log-likelihood of abrupt thaw", fontsize=7.5, labelpad=3)
    return im


def panel_b_di(fig, ax, cax, di_warp, extent, threshold):
    di_vmax = float(np.nanpercentile(di_warp, 99))
    norm = Normalize(vmin=0.0, vmax=di_vmax)
    im = ax.imshow(di_warp, extent=extent, origin="upper", interpolation="nearest",
                   cmap=DI_CMAP, norm=norm, zorder=2, rasterized=True)
    xs, ys = xy_grid(extent, di_warp.shape)
    ax.contour(xs, ys, np.where(np.isfinite(di_warp), di_warp, np.nan), levels=[threshold],
               colors="#e34a33", linewidths=0.35, alpha=0.85, zorder=2.5)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, threshold, di_vmax])
    cbar.set_ticklabels(["0", f"{threshold:.2f}", f"{di_vmax:.1f}+"])
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6, color=figstyle.INK)
    cbar.outline.set_edgecolor(figstyle.MUTED)
    cbar.outline.set_linewidth(0.35)
    cbar.set_label("Dissimilarity to training data", fontsize=7.5, labelpad=3)
    return im


def main():
    figstyle.use()
    log_evidence, di, threshold, lon, lat, valid = load_grids()
    src_tf = source_transform(lon, lat, valid)
    extent_box, (th, tw), dst_tf = dest_grid(lon, lat, valid)

    le_warp = warp_to_albers(log_evidence, src_tf, (th, tw), dst_tf)
    di_warp = warp_to_albers(di, src_tf, (th, tw), dst_tf)
    # Nearest-neighbour warp can carry a stray finite value across the off-domain
    # edge at the coarser display resolution; re-intersect the two valid masks so
    # they stay identical post-warp.
    both_valid = np.isfinite(le_warp) & np.isfinite(di_warp)
    le_warp = np.where(both_valid, le_warp, np.nan)
    di_warp = np.where(both_valid, di_warp, np.nan)

    ak = mainland_outline()
    fig = figstyle.figure("full", height=4.5, subplots=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.075], hspace=0.55, wspace=0.04,
                          left=0.02, right=0.99, top=0.97, bottom=0.11)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cax_a, cax_b = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    setup_map(ax_a, ak, extent_box)
    panel_a_susceptibility(fig, ax_a, cax_a, le_warp, extent_box)
    figstyle.panel_label(ax_a, "a")
    scale_bar(ax_a, extent_box)
    figstyle.north_arrow(ax_a, x=0.93, y=0.85, size=0.075)

    setup_map(ax_b, ak, extent_box)
    panel_b_di(fig, ax_b, cax_b, di_warp, extent_box, threshold)
    scale_bar(ax_b, extent_box)
    figstyle.north_arrow(ax_b, x=0.93, y=0.85, size=0.075)
    figstyle.panel_label(ax_b, "b")

    n_valid = int(both_valid.sum())
    n_abrupt_favoring = int((le_warp[both_valid] > 0).sum())
    n_outside_aoa = int((di_warp[both_valid] > threshold).sum())
    figstyle.save(fig, "03_susceptibility_aoa", tight=False)
    print(f"wrote 03_susceptibility_aoa.pdf/.png  "
          f"(abrupt-favoring={n_abrupt_favoring/n_valid*100:.1f}%, "
          f"outside-AOA={n_outside_aoa/n_valid*100:.1f}% of displayed valid cells)")


if __name__ == "__main__":
    main()
