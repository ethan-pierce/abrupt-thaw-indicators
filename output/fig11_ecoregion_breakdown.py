"""Figure 11 — Ecoregion breakdown (L7, §5.3 "Landscape-scale pattern").

A descriptive translation of the abrupt-thaw susceptibility surface into named
physiographic regions (EPA Level III Ecoregions of Alaska). NOT a validation and
NOT causal: it makes the statewide log-evidence>0 fraction legible in the spatial
vocabulary a permafrost scientist thinks in.

Two linked panels on one shared vik (log-evidence) scale:
  (a) a choropleth of the kept ecoregions, filled by each region's MEDIAN
      log-evidence, edged by its Level-I physiographic group, numbered 1-N;
  (b) a ranked column of per-cell log-evidence distributions (one gradient-filled
      violin per region), sorted by abrupt-favoring fraction (share of in-AOA
      cells with log-evidence > 0), with the median as a tick and the fraction
      annotated. Same 1-N key and Level-I colour tab tie each row to its polygon.

Scope: in-AOA cells only; regions with < 50% permafrost coverage are dropped
(they are majority non-permafrost, so a region-level fraction would be computed
over an unrepresentative sliver). Every kept region has >= 80% AOA coverage, so
AOA coverage is stated in the caption rather than drawn.

Class encoding (fixed): 0 = Abrupt (majority), 1 = Non-abrupt (minority).
Index is prior-free log-evidence (> 0 favors abrupt), never a probability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from affine import Affine  # noqa: F401  (documents the transform contract)
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from rasterio.features import rasterize
from scipy.stats import gaussian_kde

import figstyle
from fig02_study_area import AK_OUTLINE, scale_bar
import fig04_susceptibility_aoa as fig04

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from settings import DATA  # noqa: E402

HERE = Path(__file__).resolve().parent
ECO_SHP = DATA / "ak_eco_l3" / "ak_eco_l3.shp"
CACHE = HERE / "fig11_region_cache.npz"

DST_CRS = "EPSG:3338"          # Alaska Albers, matches Figs 2/4/10
PERM_COV_MIN = 0.50            # keep regions >= 50% permafrost coverage
VMAX = 6.0                     # symmetric log-evidence display bound

# Level-I physiographic groups -> incidental-categorical colours (Okabe-Ito,
# chosen off vik's blue<->red value axis so a group tab never reads as a value).
L1_COLOR = {
    "TUNDRA": figstyle.QUALITATIVE[1],                     # sky   #56B4E9
    "TAIGA": figstyle.QUALITATIVE[2],                      # green #009E73
    "NORTHWESTERN FORESTED MOUNTAINS": figstyle.QUALITATIVE[0],  # amber #E69F00
    "MARINE WEST COAST FOREST": figstyle.QUALITATIVE[6],   # purple #CC79A7
}
L1_LABEL = {
    "TUNDRA": "Tundra",
    "TAIGA": "Taiga",
    "NORTHWESTERN FORESTED MOUNTAINS": "NW Forested Mtns",
    "MARINE WEST COAST FOREST": "Marine West Coast Forest",
}

# Display overrides for over-long region names (row labels only).
NAME_ABBR = {
    "Interior Forested Lowlands and Uplands": "Interior Forested Lowlands/Uplands",
    "Ahklun and Kilbuck Mountains": "Ahklun & Kilbuck Mountains",
}


# --------------------------------------------------------------------------- #
# Data: assign every in-AOA datacube cell to a Level III ecoregion
# --------------------------------------------------------------------------- #
def load_cell_regions():
    """Return (log_evidence, in_AOA mask, region-code raster, code->L3name map).

    The datacube is a regular EPSG:4326 grid carrying per-cell lon/lat; recover
    its affine (fig04.source_transform) and rasterize the ecoregion polygons onto
    that exact grid, so each cell inherits the region whose polygon covers it.
    """
    le_ds = xr.open_dataset(DATA / "susceptibility.nc")
    aoa_ds = xr.open_dataset(DATA / "aoa.nc")
    log_evidence = le_ds["log_evidence"].values
    lon, lat = le_ds["longitude"].values, le_ds["latitude"].values
    inside = aoa_ds["inside_aoa"].values.astype(bool)
    valid = np.isfinite(log_evidence) & np.isfinite(lon) & np.isfinite(lat)
    ny, nx = log_evidence.shape

    eco = gpd.read_file(ECO_SHP)
    eco["geometry"] = eco.geometry.buffer(0)              # heal invalid rings
    names = sorted(eco["US_L3NAME"].unique())
    code = {n: i + 1 for i, n in enumerate(names)}        # 0 = background

    if CACHE.exists():
        region = np.load(CACHE)["region"]
    else:
        src_tf = fig04.source_transform(lon, lat, valid)
        eco4326 = eco.to_crs("EPSG:4326")
        shapes = [(g, code[n]) for g, n in zip(eco4326.geometry, eco4326["US_L3NAME"])]
        region = rasterize(shapes, out_shape=(ny, nx), transform=src_tf,
                           fill=0, dtype="int32")
        np.savez_compressed(CACHE, region=region)

    inv = {v: k for k, v in code.items()}
    return log_evidence, valid, inside, region, inv, eco


def region_stats(log_evidence, valid, inside, region, inv, eco):
    """Per-region metrics + geometries, filtered and sorted for the figure."""
    l1 = dict(zip(eco["US_L3NAME"], eco["NA_L1NAME"]))
    inaoa = valid & inside
    recs = []
    for c, name in inv.items():
        foot = region == c
        n_foot = int(foot.sum())
        n_pf = int((foot & valid).sum())
        if n_foot == 0 or n_pf == 0:
            continue
        ia = foot & inaoa
        le = log_evidence[ia]
        recs.append(dict(
            code=c, name=name, l1=l1[name],
            perm_cov=n_pf / n_foot,
            aoa_cov=int(ia.sum()) / n_pf,
            frac=float((le > 0).mean()),
            median=float(np.median(le)),
            n=int(ia.sum()),
            le=le,
        ))
    kept = [r for r in recs if r["perm_cov"] >= PERM_COV_MIN]
    kept.sort(key=lambda r: r["frac"], reverse=True)
    for i, r in enumerate(kept, 1):
        r["key"] = i

    eco_alb = eco.dissolve(by="US_L3NAME", as_index=False).to_crs(DST_CRS)
    geom = {row["US_L3NAME"]: row.geometry for _, row in eco_alb.iterrows()}
    return kept, geom, eco_alb


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _gradient_violin(ax, le, ypos, width, norm, cmap, rng, cap=40000):
    """Horizontal violin at `ypos`, filled with a log-evidence gradient.

    The fill maps screen-x (== log-evidence value) through `norm`, clipped to the
    KDE body, so the pale centre sits exactly at 0 and the warm mass to the right
    of 0 *is* the abrupt-favoring fraction, seen not just labelled.
    """
    sample = le if le.size <= cap else rng.choice(le, cap, replace=False)
    body = ax.violinplot([sample], positions=[ypos], vert=False,
                         widths=width, showextrema=False)["bodies"][0]
    body.set_facecolor("none")
    body.set_edgecolor(figstyle.MUTED)
    body.set_linewidth(0.4)
    verts = body.get_paths()[0].vertices
    x0, x1 = verts[:, 0].min(), verts[:, 0].max()
    y0, y1 = verts[:, 1].min(), verts[:, 1].max()
    grad = np.linspace(x0, x1, 256)[None, :]              # column x == LE value
    im = ax.imshow(grad, extent=[x0, x1, y0, y1], aspect="auto", cmap=cmap,
                   norm=norm, origin="lower", zorder=1, rasterized=True)
    im.set_clip_path(PathPatch(MplPath(verts), transform=ax.transData, fc="none"))


def build():
    figstyle.use()
    log_evidence, valid, inside, region, inv, eco = load_cell_regions()
    kept, geom, eco_alb = region_stats(log_evidence, valid, inside, region, inv, eco)

    used_l1 = [g for g in L1_COLOR if any(r["l1"] == g for r in kept)]
    figstyle.assert_cvd_safe([L1_COLOR[g] for g in used_l1], name="Level-I groups")

    norm = figstyle.symmetric_norm(VMAX)
    cmap = figstyle.LOG_EVIDENCE_CMAP
    rng = np.random.default_rng(42)

    fig = figstyle.figure("full", height=8.95, subplots=False)
    ax_map = fig.add_axes([0.08, 0.585, 0.84, 0.40])      # top: enlarged, centred
    cax = fig.add_axes([0.409, 0.545, 0.4125, 0.020])     # centred on violin x=0
    ax_v = fig.add_axes([0.34, 0.05, 0.55, 0.42])         # bottom: violin column

    # -- (a) choropleth (fill = median LE; no L1 outlines) ---------------- #
    ak = fig04.mainland_outline()
    eco_alb.plot(ax=ax_map, facecolor=figstyle.DOMAIN_GRAY, edgecolor="white",
                 linewidth=0.3, zorder=1)
    for r in kept:
        gpd.GeoSeries([geom[r["name"]]], crs=DST_CRS).plot(
            ax=ax_map, facecolor=cmap(norm(r["median"])),
            edgecolor="white", linewidth=0.4, zorder=2)
    ak.plot(ax=ax_map, facecolor="none", edgecolor="#666666", linewidth=0.5, zorder=3)
    for r in kept:
        pt = geom[r["name"]].representative_point()
        ax_map.annotate(str(r["key"]), (pt.x, pt.y), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", zorder=4,
                        color="white" if abs(r["median"]) > 2.6 else figstyle.INK)
    # zoom to the kept regions (drop the wasteful Aleutian/panhandle tail)
    kb = gpd.GeoSeries([geom[r["name"]] for r in kept], crs=DST_CRS).total_bounds
    pad = 120_000.0
    xmin, xmax = kb[0] - pad, kb[2] + pad
    ymin, ymax = kb[1] - pad, kb[3] + pad
    ax_map.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal",
               xticks=[], yticks=[])
    for sp in ax_map.spines.values():
        sp.set_edgecolor(figstyle.MUTED); sp.set_linewidth(0.45)
    scale_bar(ax_map, (xmin, xmax, ymin, ymax))
    figstyle.north_arrow(ax_map)
    fig.text(0.02, 0.975, "(a)", fontsize=11, fontweight="bold",
             color=figstyle.INK, va="top", ha="left")

    # -- shared log-evidence colorbar: horizontal, centred on the violins'
    #    zero line, above the violin column -------------------------------- #
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([-VMAX, 0, VMAX])
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_title("Log-evidence", fontsize=10.5, pad=4)
    cbar.ax.text(0.0, -2.6, "non-abrupt", transform=cbar.ax.transAxes,
                 ha="left", va="top", fontsize=9, color=figstyle.INK)
    cbar.ax.text(1.0, -2.6, "abrupt", transform=cbar.ax.transAxes,
                 ha="right", va="top", fontsize=9, color=figstyle.INK)

    # -- Level-I physiographic-group legend (stacked, above panel b, left) - #
    fig.legend(handles=[Patch(facecolor=L1_COLOR[g], edgecolor="none",
                              label=L1_LABEL[g]) for g in used_l1],
               title="Physiographic group (Level I)", loc="upper left",
               bbox_to_anchor=(0.02, 0.575), ncol=1, fontsize=9,
               title_fontsize=9.5, frameon=False, handlelength=1.2,
               handleheight=1.2, labelspacing=0.5)

    # -- (b) ranked gradient violins -------------------------------------- #
    ypos = np.arange(len(kept))[::-1]
    ytr = ax_v.get_yaxis_transform()                      # x: axes, y: data
    for r, yi in zip(kept, ypos):
        _gradient_violin(ax_v, r["le"], yi, 0.82, norm, cmap, rng)
        ax_v.plot([r["median"]], [yi], marker="|", ms=13, mew=3.0,
                  color="white", zorder=3)                          # halo
        ax_v.plot([r["median"]], [yi], marker="|", ms=11, mew=1.4,
                  color=figstyle.INK, zorder=3.1)
        ax_v.plot(-0.02, yi, marker="s", ms=8, transform=ytr, clip_on=False,
                  color=L1_COLOR[r["l1"]], zorder=4)                 # L1 tab
        ax_v.text(1.015, yi, f'{r["frac"] * 100:.0f}%', transform=ytr,
                  va="center", ha="left", fontsize=9, color=figstyle.INK)
    ax_v.axvline(0, color=figstyle.INK, lw=0.9, zorder=2)
    ax_v.set_yticks(ypos)
    ax_v.set_yticklabels([f'{r["key"]}. {NAME_ABBR.get(r["name"], r["name"])}'
                          for r in kept], fontsize=9)
    ax_v.tick_params(axis="y", length=0, pad=16)          # room for the L1 tab
    ax_v.set_ylim(-0.7, len(kept) - 0.3)
    ax_v.set_xlim(-VMAX - 2, VMAX + 2)
    ax_v.set_xlabel("Log-evidence", fontsize=10.5)
    ax_v.tick_params(axis="x", labelsize=9)
    ax_v.text(1.015, len(kept) - 0.1, "Ratio of cells\nfavoring\nabrupt thaw",
              transform=ytr, va="bottom", ha="left", fontsize=8.5,
              color=figstyle.MUTED)
    ax_v.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.02, 0.60, "(b)", fontsize=11, fontweight="bold",
             color=figstyle.INK, va="top", ha="left")

    figstyle.save(fig, "11_ecoregion_breakdown", tight=False)
    return fig


if __name__ == "__main__":
    build()
    print("wrote 11_ecoregion_breakdown.{pdf,png}")
