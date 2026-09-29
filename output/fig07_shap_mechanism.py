"""Figure 7 — SHAP dependence for the top-9 continuous indicators.

Each panel plots a feature's own SHAP against its own value, for the top-9
continuous features by mean |SHAP|. Land Cover is omitted because a one-hot has no continuous shape; it is Fig 6b.
Reads output/shap_mechanism_cache.npz (out-of-fold SHAP, positive favors Abrupt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import figstyle  # noqa: E402
import shap_family_display as fd  # noqa: E402

CACHE = _HERE / "shap_mechanism_cache.npz"

WARM = figstyle.ABRUPT          # own SHAP > 0  -> favors Abrupt
COOL = figstyle.NON_ABRUPT      # own SHAP < 0  -> favors Non-abrupt

N_XBINS = 22                    # fixed-width trend bins across the displayed x-range
BIN_N_MIN = 40                  # min points per bin to draw a trend node (drops sparse tails)

# A shared symmetric y-range is set by the temperature warm-edge tail and flattens
# every other panel, so each panel gets its own robust range by default.
SHARED_Y = False

# Top-9 continuous features by own mean|SHAP|, in reading order.
# scale rescales the stored value to display units; logx for heavy tails.
ROSTER = [
    dict(name="Slope", title="Slope", unit="°",
         scale=1.0, floor0=True, slope_mass=True),
    dict(name="Annual Mean Temperature", title="Mean Annual Air Temperature", unit="°C",
         scale=0.1, warm_cliff=-4.0),                 # WorldClim V1 stores tenths of °C
    dict(name="Trend in SWE", title="Trend in SWE", unit="mm yr$^{-1}$",
         scale=1.0),
    dict(name="Isothermality", title="Isothermality", unit="%",
         scale=1.0),
    dict(name="Mean Annual SWE", title="Mean Annual SWE", unit="mm",
         scale=1.0),
    dict(name="Height Above Nearest Drainage", title="Height above drainage", unit="m",
         scale=1.0),
    dict(name="Trend in precipitation", title="Trend in precipitation", unit="mm yr$^{-1}$",
         scale=1.0),
    dict(name="Mean curvature (500 m)", title="Mean curvature (500 m)", unit="×10$^{-3}$",
         scale=1000.0),
    dict(name="Upstream Area", title="Upstream area", unit="km$^2$",
         scale=1.0, logx=True),                       # heavy right tail -> log x
]


def load():
    d = np.load(CACHE, allow_pickle=True)
    names = list(d["feature_names"])
    return names, d["values"].astype(float), d["data"].astype(float)


def trend(x, g, xlo, xhi, *, logx=False):
    """Running median + 25/75 band over fixed-width x-bins with a min-count floor."""
    edges = (np.geomspace(xlo, xhi, N_XBINS + 1) if logx
             else np.linspace(xlo, xhi, N_XBINS + 1))
    xc, med, q1, q3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < BIN_N_MIN:
            continue
        xc.append(float(np.median(x[m])))
        lo, mid, hi = np.percentile(g[m], [25, 50, 75])
        med.append(mid); q1.append(lo); q3.append(hi)
    return np.array(xc), np.array(med), np.array(q1), np.array(q3)


def draw_trend(ax, xc, med, q1, q3):
    """Sign-colored running-median line."""
    if len(xc) < 2:
        return
    pts = np.array([xc, med]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    mid = 0.5 * (med[:-1] + med[1:])
    cols = [WARM if m >= 0 else COOL for m in mid]
    ax.add_collection(LineCollection(segs, colors=cols, lw=2.4, zorder=4,
                                     capstyle="round"))


def panel(ax, spec, x_raw, g, *, ylim):
    ok = np.isfinite(x_raw) & np.isfinite(g)              # drop rows where the feature is NaN
    x, g = x_raw[ok] * spec["scale"], g[ok]
    logx = spec.get("logx", False)

    xlo, xhi = np.nanpercentile(x, [0.5, 99.5])
    if spec.get("floor0"):
        xlo = 0.0                                        # genuine floor (flat ground)

    inrange = (x >= xlo) & (x <= xhi)
    xc_x, xc_g = x[inrange], g[inrange]                  # bound the hexbin binning region

    # Display left edge: for the slope panel, pad left of 0 so the big x=0 point
    # mass (and its marker) is not sliced in half by the spine.
    x_left = xlo - 0.035 * (xhi - xlo) if spec.get("slope_mass") else xlo

    if logx:
        ax.set_xscale("log")
        ax.hexbin(xc_x, xc_g, gridsize=(30, 24), xscale="log", bins="log", mincnt=1,
                  cmap="Greys", linewidths=0, zorder=1, rasterized=True)
        ax.set_xlim(xlo, xhi)
    else:
        ax.hexbin(xc_x, xc_g, gridsize=(30, 24), extent=(x_left, xhi, ylim[0], ylim[1]),
                  bins="log", mincnt=1, cmap="Greys", linewidths=0, zorder=1,
                  rasterized=True)
        ax.set_xlim(x_left, xhi)
    ax.set_ylim(*ylim)
    ax.axhline(0, color=figstyle.INK, lw=0.7, zorder=2)

    # Warm-cliff: shade the sampling-/boundary-sensitive warm-edge region (no text).
    if "warm_cliff" in spec:
        wc = spec["warm_cliff"]
        if wc < xhi:
            ax.axvspan(wc, xhi, color=COOL, alpha=0.10, lw=0, zorder=0)

    # Slope-0 point mass: peeled out as a distinct marker; trend starts at the
    # smallest positive slope bin so the continuous gradient is legible.
    if spec.get("slope_mass"):
        zero = x == 0
        lo, mid, hi = np.percentile(g[zero], [25, 50, 75])
        ax.errorbar([0], [mid], yerr=[[mid - lo], [hi - mid]], fmt="D", ms=6.5,
                    mfc=WARM, mec="white", mew=1.0, ecolor=figstyle.INK,
                    elinewidth=1.1, capsize=2.6, zorder=6)
        ax.annotate(f"0°: {zero.mean() * 100:.0f}% of points", xy=(0, mid),
                    xytext=(0.30, 0.90), textcoords="axes fraction", fontsize=8,
                    color=figstyle.INK, va="top",
                    arrowprops=dict(arrowstyle="-", color=figstyle.MUTED, lw=0.8))
        pos = x > 0
        tlo = np.nanpercentile(x[pos], 0.5)
        draw_trend(ax, *trend(x[pos], g[pos], tlo, xhi))
    else:
        draw_trend(ax, *trend(x, g, xlo, xhi, logx=logx))

    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    figstyle.use()
    names, values, data = load()

    def own(name):
        col = values[:, names.index(name)]
        return col, float(np.mean(np.abs(col[np.isfinite(col)])))

    imps = [own(s["name"])[1] for s in ROSTER]
    families = fd.family_of_features([s["name"] for s in ROSTER])

    # shared symmetric y from robust extremes across the 9 own-SHAP arrays
    cols = [values[:, names.index(s["name"])] for s in ROSTER]
    cols = [c[np.isfinite(c)] for c in cols]
    ymax = max(max(abs(np.percentile(c, 0.5)), abs(np.percentile(c, 99.5))) for c in cols)
    shared_ylim = (-ymax, ymax)

    w = figstyle.WIDTHS_IN["full"]
    fig = plt.figure(figsize=(w, 7.7))
    gs = GridSpec(3, 3, figure=fig, wspace=0.14 if SHARED_Y else 0.30, hspace=0.62,
                  left=0.165, right=0.985, top=0.925, bottom=0.06)

    for j, spec in enumerate(ROSTER):
        r, c = divmod(j, 3)
        ax = fig.add_subplot(gs[r, c])
        col = values[:, names.index(spec["name"])]
        x_raw = data[:, names.index(spec["name"])]

        if SHARED_Y:
            ylim = shared_ylim
        else:
            # Asymmetric robust range, always including 0.
            cc = col[np.isfinite(col)]
            lo, hi = np.percentile(cc, [0.5, 99.5])
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            pad = 0.06 * (hi - lo)
            ylim = (lo - pad, hi + pad)

        panel(ax, spec, x_raw, col, ylim=ylim)

        if SHARED_Y and c != 0:
            ax.set_yticklabels([])
        if c == 0:
            ax.set_ylabel("SHAP  (margin)", fontsize=9.5)
            ax.text(-0.46, 0.985, "Abrupt", transform=ax.transAxes, rotation=90,
                    ha="center", va="top", fontsize=8.5, color=WARM, fontweight="bold")
            ax.text(-0.46, 0.015, "Non-abrupt", transform=ax.transAxes, rotation=90,
                    ha="center", va="bottom", fontsize=8.5, color=COOL, fontweight="bold")
        ax.set_xlabel(spec["unit"], fontsize=9)
        family = families[j]
        ax.set_title(f"{j + 1}. {spec['title']}", fontsize=10.5, color=figstyle.INK,
                     pad=16)
        ax.text(0.5, 1.03, fd.NAMES[family], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8, color=figstyle.MUTED)

    figstyle.save(fig, "07_shap_mechanism", outdir=_HERE, tight=False)
    tag = f"shared y = ±{ymax:.2f}" if SHARED_Y else "per-panel y (fallback)"
    print(f"Wrote 07_shap_mechanism.{{pdf,png}} | {tag}")
    print("Roster (own mean|SHAP|, rank):")
    for j, spec in enumerate(ROSTER):
        print(f"  {j + 1}. {spec['name']:34s} {imps[j]:.3f}"
              + ("  [logx]" if spec.get("logx") else ""))


if __name__ == "__main__":
    main()
