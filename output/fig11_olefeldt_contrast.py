"""Figure 11 — Olefeldt incumbent contrast (L7, Form B).

Positioning against the only Alaska-statewide comparable incumbent, Olefeldt
et al. (2016) thermokarst-landscape classes. Our log-evidence index measures
thaw *mode* (abrupt vs non-abrupt); Olefeldt maps thermokarst *occurrence
potential + landform type*. The figure shows these are a **largely orthogonal
axis**: within every Olefeldt thermokarst class the log-evidence spans the full
range, and Olefeldt class explains only ~1-7% of the variation in mode (per-type
eta-squared, annotated). The lone weak, mechanistically-sensible lean is
Hillslope (retrogressive thaw slumps *are* an abrupt form) creeping toward the
neutral line as its rated potential rises. Positioning, NOT validation — there
is no statewide mode ground truth.

Design (raincloud, one shared log-evidence axis, None dropped):
  * three thermokarst types (Wetland / Lake / Hillslope), each split by ordinal
    potential Low -> Very High (Olefeldt's "None" is dropped: under per-type
    faceting it is a cross-contaminated grab-bag — "no *wetland* thermokarst"
    includes strong-*hillslope* cells — so it is reported in the caption, not
    plotted);
  * per class: a half-violin "cloud" (KDE of log-evidence), a subsampled point
    "rain" (actual in-AOA cells), and a single median marker. No box-and-whisker
    (redundant with the cloud, and it hides the very data this figure shows);
  * dashed reference at the all-cell in-AOA median (-2.50): every class median
    clings to it -> class barely moves mode;
  * warm above 0 / cool below (house law: warm = abrupt).

The polygon->grid join (Olefeldt LAEA polygons rasterized onto the datacube's
EPSG:4326 grid, restricted to in-AOA cells) is heavy, so it is cached to
output/fig11_olefeldt_cache.npz on first run. Reads data/susceptibility.nc +
data/aoa.nc + data/Circumpolar_Thermokarst_Landscapes/. Writes
output/11_olefeldt_contrast.{pdf,png}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import figstyle  # noqa: E402

REPO = _HERE.parent
DATA = REPO / "data"
SUSCEPTIBILITY_NC = DATA / "susceptibility.nc"
AOA_NC = DATA / "aoa.nc"
OLEFELDT_SHP = DATA / "Circumpolar_Thermokarst_Landscapes/Circumpolar_Thermokarst_Landscapes.shp"
CACHE = _HERE / "fig11_olefeldt_cache.npz"

# Olefeldt LAEA (ESRI:102017) -> datacube EPSG:4326.
OLEFELDT_CRS = "ESRI:102017"
GRID_CRS = "EPSG:4326"

# level code: 0 = outside Olefeldt coverage, 1 = None, 2..5 = Low..Very High
LEVELS = {"None": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}
LEVEL_LABEL = {2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}

# type field -> (display name, hue). Wetland teal-green / Lake blue / Hillslope
# orange: an intuitive, CVD-safe Okabe-Ito triple (checked in main()).
TYPES = [
    ("TKWP", "lw", "Wetland", "#009E73"),
    ("TKThLP", "ll", "Lake", "#0072B2"),
    ("TKHP", "lh", "Hillslope", "#D55E00"),
]

# raincloud layout
STEP, GROUP_GAP = 1.9, 1.3      # row pitch within a type / gap between types
CLOUD_H, RAIN_H = 0.85, 0.80    # half-violin height above baseline / rain depth below
XLIM = (-14.0, 4.2)
N_RAIN = 1500                   # points drawn per class ("rain")
N_KDE = 40000                   # cells subsampled for each KDE (speed)


def build_cache() -> None:
    """Rasterize the three Olefeldt potential layers onto the in-AOA grid."""
    import xarray as xr
    import geopandas as gpd
    from affine import Affine
    from rasterio.features import rasterize
    from pyproj import Transformer

    le = xr.open_dataset(SUSCEPTIBILITY_NC)
    aoa = xr.open_dataset(AOA_NC)
    log_ev = le["log_evidence"].values
    di = aoa["DI"].values
    thr = float(aoa.attrs["aoa_threshold"])
    lon, lat = le["longitude"].values, le["latitude"].values
    valid = np.isfinite(log_ev) & np.isfinite(lon) & np.isfinite(lat)
    if not np.array_equal(valid, np.isfinite(di)):
        raise ValueError("susceptibility.nc and aoa.nc must share an identical valid mask")
    in_aoa = valid & (di <= thr)

    # recover the datacube's regular EPSG:4326 affine (same fit as fig03)
    ys, xs = np.nonzero(valid)
    dlon, lon0 = np.polyfit(xs, lon[ys, xs], 1)
    dlat, lat0 = np.polyfit(ys, lat[ys, xs], 1)
    tf = Affine(dlon, 0.0, lon0 - dlon / 2.0, 0.0, dlat, lat0 - dlat / 2.0)
    H, W = log_ev.shape

    # read only the Alaska footprint of the circumpolar shapefile
    lo0, lo1 = float(np.nanmin(lon[valid])), float(np.nanmax(lon[valid]))
    la0, la1 = float(np.nanmin(lat[valid])), float(np.nanmax(lat[valid]))
    t = Transformer.from_crs(GRID_CRS, OLEFELDT_CRS, always_xy=True)
    bx, by = t.transform([lo0, lo1, lo0, lo1], [la0, la0, la1, la1])
    g = gpd.read_file(OLEFELDT_SHP, bbox=(min(bx), min(by), max(bx), max(by))).to_crs(GRID_CRS)

    def burn(field):
        codes = g[field].map(LEVELS).fillna(0).astype("uint8")
        return rasterize(((geom, c) for geom, c in zip(g.geometry, codes)),
                         out_shape=(H, W), transform=tf, fill=0, dtype="uint8")

    layers = {slug: burn(field)[in_aoa] for field, slug, _, _ in TYPES}
    np.savez_compressed(
        CACHE,
        y=log_ev[in_aoa].astype("float32"),
        overall_median=np.float64(np.median(log_ev[in_aoa])),
        n_in_aoa=np.int64(in_aoa.sum()),
        **layers,
    )
    print(f"cached {int(in_aoa.sum()):,} in-AOA cells -> {CACHE.name}")


def load_cache():
    if not CACHE.exists():
        build_cache()
    d = np.load(CACHE)
    return d


def _eta2(levels, y):
    """Share of log-evidence variance explained by potential class (None dropped)."""
    m = levels >= 2
    grand = y[m].mean()
    grps = [y[levels == c] for c in range(2, 6) if (levels == c).sum() > 0]
    ssb = sum(len(gp) * (gp.mean() - grand) ** 2 for gp in grps)
    return ssb / ((y[m] - grand) ** 2).sum()


def _nfmt(n):
    return f"{n / 1000:.0f}k" if n >= 1000 else str(int(n))


def render(d):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    figstyle.use()
    y = d["y"]
    overall_med = float(d["overall_median"])
    rng = np.random.default_rng(42)
    xgrid = np.linspace(XLIM[0], XLIM[1], 320)

    fig, ax = plt.subplots(figsize=(figstyle.WIDTHS_IN["onehalf"], 6.8))
    ax.axvspan(0, XLIM[1], color=to_rgb(figstyle.ABRUPT), alpha=0.05, zorder=0)
    ax.axvspan(XLIM[0], 0, color=to_rgb(figstyle.NON_ABRUPT), alpha=0.05, zorder=0)
    ax.axvline(0, color=figstyle.INK, lw=0.9, zorder=2)
    ax.axvline(overall_med, color=figstyle.MUTED, lw=0.8, ls=(0, (2, 2)), zorder=2)

    row = 0.0
    yticks, ylabels = [], []
    for _field, slug, name, col in TYPES:
        levels = d[slug]
        present = [c for c in range(2, 6) if (levels == c).sum() > 0]
        rgb = np.array(to_rgb(col))
        light = tuple(1 - 0.55 * (1 - rgb))
        grp_rows = []
        for c in present:
            v = y[levels == c]
            sub = v if len(v) <= N_KDE else rng.choice(v, N_KDE, replace=False)
            dens = gaussian_kde(sub, bw_method=0.28)(xgrid)
            dens = dens / dens.max() * CLOUD_H
            ax.fill_between(xgrid, row, row + dens, color=light, alpha=0.55, lw=0, zorder=3)
            ax.plot(xgrid, row + dens, color=col, lw=0.7, alpha=0.85, zorder=4)
            rain = v if len(v) <= N_RAIN else rng.choice(v, N_RAIN, replace=False)
            ax.scatter(rain, row - rng.uniform(0.12, RAIN_H, len(rain)), s=1.5, color=light,
                       alpha=0.30, lw=0, zorder=3, rasterized=True)
            ax.scatter([np.median(v)], [row], color="white", edgecolor=col, s=34, lw=1.5, zorder=7)
            ax.text(XLIM[1] - 0.15, row + 0.68, f"n = {_nfmt(len(v))}", fontsize=6.8,
                    color=figstyle.INK, ha="right", va="center")
            yticks.append(row)
            ylabels.append(LEVEL_LABEL[c])
            grp_rows.append(row)
            row += STEP
        # centre the type label vertically, nudged into a between-row gap for
        # odd row counts (Hillslope) so it never sits on a level baseline
        label_y = float(np.mean(grp_rows))
        if len(grp_rows) % 2 == 1:
            label_y += STEP / 2.0
        ax.text(XLIM[0] + 0.3, label_y, name, rotation=0, va="center",
                ha="left", fontsize=10, color=col, fontweight="bold")
        ax.text(XLIM[0] + 0.3, max(grp_rows) + CLOUD_H + 0.15, f"$\\eta^2$ {100 * _eta2(levels, y):.0f}%",
                fontsize=8, color=col, ha="left", va="top", fontweight="bold")
        row += GROUP_GAP

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-RAIN_H - 0.4, row - GROUP_GAP + CLOUD_H + 0.3)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Log-evidence  (← non-abrupt  ·  abrupt →)", fontsize=9.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def main():
    d = load_cache()
    # accessibility gate: the three type hues must stay distinct under CVD
    try:
        figstyle.assert_cvd_safe([c for *_, c in TYPES], min_de=15, name="Fig11 type hues")
        print("CVD check: type hues OK (min ΔE ≥ 15)")
    except Exception as exc:  # noqa: BLE001
        print(f"CVD check WARNING: {exc}")
    fig = render(d)
    pdf = figstyle.save(fig, "11_olefeldt_contrast")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
