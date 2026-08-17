"""Figure 9 — SHAP dominant-domain map (L6c / L7).

For every in-AOA grid cell, which thematic *domain* moves the model's prediction
most? Per cell we summed each domain's per-feature SHAP to a net contribution and
took the domain with the largest |net| (unsigned — which KIND of driver dominates,
not its direction; direction is Fig 3's job). This map paints that dominant
domain, hue only (winner-take-all; no saturation/alpha modulation).

Its argumentative job is the PROXY REBUTTAL (§5.2): the dominant driver varies
regionally, so the signal is not one smooth spatial trend a single location-proxy
would produce. Descriptive only — all-data model, AOA-restricted display, no
calibrated-probability or causal claim.

The companion area-fraction bars quantify the map (% of in-AOA area each domain
dominates) and carry the honesty about near-ties (winner-take-all hides how close
the runner-up was); the VALIDATION GATE (a single domain > 60% of area) is checked
in the cache script — here Baseline temperature leads at 45%, so the map stays polychrome.

Pure plotting. The heavy per-cell TreeSHAP lives in
models/shap_dominance_cache.py -> output/shap_dominance_cache.npz. Domain palette
+ feature->domain assignment come from output/shap_domains.py (shared with Fig 6).
Map warp (EPSG:4326 -> Alaska Albers) reuses output/fig03_thaw_mode_and_aoa.py.

Writes output/09_shap_dominance.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from pyproj import Transformer
from rasterio.enums import Resampling

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import figstyle  # noqa: E402
import shap_domains as sd  # noqa: E402
import fig03_thaw_mode_and_aoa as fig03  # noqa: E402  (reuse the warp + basemap)

CACHE = _HERE / "shap_dominance_cache.npz"
SUSCEPTIBILITY_NC = fig03.SUSCEPTIBILITY_NC   # same grid as the dominance raster

DISPLAY_MIN = 0.01          # domains below this in-AOA share fold into "Other"
BACKDROP_GRAY = "#e1e4e6"   # in-permafrost-domain but out-of-AOA (Fig 2 domain fill)
OTHER_LABEL = "Other"


def load_cache():
    d = np.load(CACHE, allow_pickle=True)
    return (d["dominant_domain"], list(d["domains"]), d["area_fraction"],
            str(d["gate_top_domain"]), float(d["gate_top_fraction"]), bool(d["gate_ok"]))


def load_grid():
    """lon/lat + permafrost-domain valid mask on the datacube grid (matches cache)."""
    s = xr.open_dataset(SUSCEPTIBILITY_NC)
    le = s["log_evidence"].values
    lon, lat = s["longitude"].values, s["latitude"].values
    valid = np.isfinite(le)                                  # the Obu permafrost domain
    coord_ok = valid & np.isfinite(lon) & np.isfinite(lat)   # for the affine regression
    return lon, lat, valid, coord_ok


def build_display_codes(dom_raster, domains, fracs):
    """Coded raster + the ordered display domains, Other fold, and inset shares.

    Codes: 0..K-1 displayed domains (share desc); K = Other (folded small domains,
    in-AOA); K+1 = out-of-AOA-but-in-domain backdrop; NaN = ocean / off-domain.
    """
    order = np.argsort(fracs)[::-1]
    disp = [domains[i] for i in order if fracs[i] >= DISPLAY_MIN]
    disp_idx = {d: i for i, d in enumerate(disp)}
    K = len(disp)
    other_code, mask_code = K, K + 1

    # raster domain code (0..7, = DOMAIN_ORDER index) -> display code
    dom_to_disp = np.array([disp_idx.get(d, other_code) for d in domains], dtype=int)

    return disp, disp_idx, K, other_code, mask_code, dom_to_disp, order


def coded_raster(dom_raster, valid, dom_to_disp, mask_code):
    ny, nx = dom_raster.shape
    coded = np.full((ny, nx), np.nan, dtype="float64")
    in_aoa = dom_raster >= 0
    coded[valid & ~in_aoa] = mask_code                       # reliability backdrop
    ys, xs = np.nonzero(in_aoa)
    coded[ys, xs] = dom_to_disp[dom_raster[ys, xs]]          # attributed domain / Other
    return coded


def to_rgba(codes_warp, disp, other_code, mask_code):
    """Categorical RGBA image from the warped integer codes."""
    palette = [sd.DOMAIN_COLORS[d] for d in disp] + [figstyle.OTHER_GRAY, BACKDROP_GRAY]
    th, tw = codes_warp.shape
    rgba = np.zeros((th, tw, 4), dtype="float32")
    ci = np.round(codes_warp)
    for code, col in enumerate(palette):
        m = ci == code
        rgba[m, :3] = to_rgb(col)
        rgba[m, 3] = 1.0
    return rgba


def graticule_labels(ax, extent):
    """Subtle edge labels for the fig02 graticule (meridians -165/-155/-145°W,
    parallels 60/65/70°N). Placed where each line crosses the frame — bottom for
    meridians, left for parallels — so the journal has its lon/lat reference."""
    xmin, xmax, ymin, ymax = extent
    tf = Transformer.from_crs("EPSG:4326", fig03.DST_CRS, always_xy=True)
    lat_s = np.linspace(45.0, 78.0, 500)
    lon_s = np.linspace(-185.0, -125.0, 500)
    kw = dict(fontsize=5.5, color=figstyle.MUTED, zorder=6, clip_on=False)

    for lon in (-165, -155, -145):                          # meridians -> bottom edge
        gx, gy = tf.transform(np.full_like(lat_s, float(lon)), lat_s)
        o = np.argsort(gy)
        if gy[o].min() <= ymin <= gy[o].max():
            x_at = float(np.interp(ymin, gy[o], gx[o]))
            if xmin <= x_at <= xmax:
                ax.annotate(f"{abs(lon)}°W", xy=(x_at, ymin), xytext=(0, -2),
                            textcoords="offset points", ha="center", va="top", **kw)

    for lat in (60, 65, 70):                                # parallels -> left edge
        gx, gy = tf.transform(lon_s, np.full_like(lon_s, float(lat)))
        o = np.argsort(gx)
        if gx[o].min() <= xmin <= gx[o].max():
            y_at = float(np.interp(xmin, gx[o], gy[o]))
            if ymin <= y_at <= ymax:
                ax.annotate(f"{lat}°N", xy=(xmin, y_at), xytext=(-2, 0),
                            textcoords="offset points", ha="right", va="center", **kw)


def area_fraction_panel(ax, disp, disp_idx, fracs, domains, order):
    """Horizontal bars: % of in-AOA area each displayed domain (+ Other) dominates."""
    shares = [float(fracs[domains.index(d)]) for d in disp]
    other_share = float(sum(fracs[i] for i in order if domains[i] not in disp_idx))
    labels = list(disp)
    colors = [sd.DOMAIN_COLORS[d] for d in disp]
    if other_share > 0:
        labels.append(OTHER_LABEL)
        shares.append(other_share)
        colors.append(figstyle.OTHER_GRAY)

    y = np.arange(len(labels))[::-1]                         # largest at top
    ax.barh(y, np.array(shares) * 100, height=0.74, color=colors, zorder=2,
            edgecolor=figstyle.INK, linewidth=0.3)
    for yi, s, lab in zip(y, shares, labels):
        ax.annotate(f"{s*100:.0f}%" if s * 100 >= 0.5 else "<1%",
                    xy=(s * 100, yi), xytext=(3, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6.5, color=figstyle.MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(0, max(shares) * 100 * 1.18)
    ax.set_xlabel("% of in-AOA area", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_title("% area where feature\ngroup dominates", fontsize=7.6,
                 color=figstyle.INK, loc="left", pad=6)


def main():
    figstyle.use()
    dom_raster, domains, fracs, gate_dom, gate_frac, gate_ok = load_cache()
    lon, lat, valid, coord_ok = load_grid()

    disp, disp_idx, K, other_code, mask_code, dom_to_disp, order = \
        build_display_codes(dom_raster, domains, fracs)

    # CVD gate over the domains actually rendered as spatial fills (spec).
    figstyle.assert_cvd_safe([sd.DOMAIN_COLORS[d] for d in disp], min_de=15,
                             name="Fig 9 displayed-domain palette")

    coded = coded_raster(dom_raster, valid, dom_to_disp, mask_code)

    # Warp the EPSG:4326 coded raster into Alaska Albers (reuse Fig 3's machinery);
    # NEAREST is mandatory — these are class codes, not a continuous field.
    src_tf = fig03.source_transform(lon, lat, coord_ok)
    extent_box, (th, tw), dst_tf = fig03.dest_grid(lon, lat, coord_ok)
    codes_warp = fig03.warp_to_albers(coded, src_tf, (th, tw), dst_tf,
                                      resampling=Resampling.nearest)
    rgba = to_rgba(codes_warp, disp, other_code, mask_code)

    ak = fig03.mainland_outline()
    fig = figstyle.figure("full", height=4.35, subplots=False)
    # Two gridspecs so the bar panel can sit lower (headroom for its two-line
    # title) while the map still uses the full figure height.
    gs_map = fig.add_gridspec(1, 1, left=0.005, right=0.70, top=0.99, bottom=0.055)
    gs_bar = fig.add_gridspec(1, 1, left=0.775, right=0.975, top=0.86, bottom=0.155)
    ax_map = fig.add_subplot(gs_map[0, 0])
    ax_bar = fig.add_subplot(gs_bar[0, 0])

    fig03.setup_map(ax_map, ak, extent_box)
    ax_map.imshow(rgba, extent=extent_box, origin="upper", interpolation="nearest",
                  zorder=2, rasterized=True)
    fig03.scale_bar(ax_map, extent_box)
    figstyle.north_arrow(ax_map, x=0.95, y=0.84, size=0.07)
    graticule_labels(ax_map, extent_box)

    # backdrop key (out-of-AOA reliability sliver) — a small note in the map corner
    handles = [Patch(facecolor=BACKDROP_GRAY, edgecolor=figstyle.MUTED, linewidth=0.3,
                     label="in domain, outside AOA")]
    ax_map.legend(handles=handles, loc="lower left", fontsize=6.3, frameon=False,
                  handlelength=1.1, handleheight=1.0, borderpad=0.4,
                  bbox_to_anchor=(0.015, 0.015))

    area_fraction_panel(ax_bar, disp, disp_idx, fracs, domains, order)

    figstyle.save(fig, "09_shap_dominance", outdir=_HERE, tight=False)
    print(f"wrote 09_shap_dominance.{{pdf,png}}  | displayed domains: {disp} "
          f"(+Other) | gate: {gate_dom} {gate_frac*100:.1f}% "
          f"{'PASS' if gate_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
