"""Figure 7 — Indicator-family SHAP: magnitude + signed direction (L6a + L6b).

Opens the interpretation section by reporting SHAP over the emergent feature
*families* (not the ~70 partly-redundant columns), so credit is not split across
near-duplicate features. Two panels sharing one importance-sorted family axis:

  (a) magnitude  — grouped importance, mean over points of |sum of member SHAP|
                   (margin). Answers "which families matter", unsigned.
  (b) direction  — the per-point signed contribution distribution per family,
                   drawn as a zero-split violin (mass right of 0 warm = favors
                   Abrupt, left cool = favors Non-abrupt), KDE clipped to the
                   observed data range, with median + 5/95 marks overlaid.

Both are needed: Land Cover is #3 by magnitude yet its signed distribution
straddles zero (it discriminates in *both* directions) — a mean-signed bar would
erase it, the violin shows it.

The family construction (feature-space Spearman clustering, NOT SHAP-space — the
anti-circularity point) is reported by the dendrogram, which lives in the
Supplement (output/shap_family_dendrogram.png); this figure states it in prose /
caption only.

Data: output/shap_grouped_matrix.npz (per-point grouped-SHAP matrix, columns
importance-sorted), written by models/shap_groups.py. Regenerate that cache
(a multi-minute OOF fold-refit TreeSHAP run) if the model or feature set changes;
this script is pure plotting.

Writes output/07_shap_families.{pdf,png}.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import figstyle  # noqa: E402

CACHE = _HERE / "shap_grouped_matrix.npz"

BAR_COLOR = "#1b6b6b"            # deep desaturated teal — off the warm/cool axis, not Fig 6 green
WARM = figstyle.ABRUPT          # >0 favors Abrupt
COOL = figstyle.NON_ABRUPT      # <0 favors Non-abrupt

VIOLIN_HALF = 0.40              # half-height of a violin body (row spacing is 1.0)
KDE_POINTS = 256


def load_cache():
    """Return (labels, importance, G) with columns already importance-sorted desc."""
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} not found — run `poetry run python models/shap_groups.py` first "
            "to write the per-point grouped-SHAP matrix.")
    d = np.load(CACHE, allow_pickle=True)
    return list(d["labels"]), d["importance"].astype(float), d["G"].astype(float)


def magnitude_panel(ax, ys, labels, importance):
    """(a) horizontal teal magnitude bars; family names on this (shared) axis."""
    frac = importance / importance.sum() * 100.0
    ax.barh(ys, importance, height=0.72, color=BAR_COLOR, zorder=2)
    for y, imp, f in zip(ys, importance, frac):
        # % of summed family importance — the share the narrative quotes.
        ax.annotate(f"{f:.0f}%" if f >= 0.5 else "<1%",
                    xy=(imp, y), xytext=(3, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6.5, color=figstyle.MUTED)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(0, importance.max() * 1.14)
    ax.set_xlabel("Mean |Σ member SHAP|  (margin)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.0, 1.012, "(a)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=figstyle.INK)


def _violin(ax, vals, y, xlim):
    """One zero-split violin at row `y`: warm right of 0, cool left; marks overlaid."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    med = float(np.median(vals))
    p5, p95 = np.percentile(vals, [5, 95])

    lo, hi = float(vals.min()), float(vals.max())     # clip KDE support to observed range
    drew_body = False
    if hi - lo > 1e-9:
        try:
            xs = np.linspace(lo, hi, KDE_POINTS)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                d = gaussian_kde(vals)(xs)
            d = d / d.max() * VIOLIN_HALF
            # split fill at exactly x=0 so no wedge is miscolored at the seam
            xs_f = np.concatenate([xs, [0.0]])
            d_f = np.concatenate([d, [np.interp(0.0, xs, d)]])
            o = np.argsort(xs_f); xs_f, d_f = xs_f[o], d_f[o]
            ax.fill_between(xs_f, y - d_f, y + d_f, where=xs_f >= 0,
                            color=WARM, alpha=0.85, lw=0, zorder=2, interpolate=True)
            ax.fill_between(xs_f, y - d_f, y + d_f, where=xs_f <= 0,
                            color=COOL, alpha=0.85, lw=0, zorder=2, interpolate=True)
            drew_body = True
        except np.linalg.LinAlgError:
            pass

    # honest quantile overlay: 5-95 whisker + median tick, on top of the silhouette
    ax.plot([max(p5, xlim[0]), min(p95, xlim[1])], [y, y],
            color=figstyle.INK, lw=0.8, zorder=4, solid_capstyle="butt")
    ax.plot([med, med], [y - VIOLIN_HALF * 0.72, y + VIOLIN_HALF * 0.72],
            color="white" if drew_body else figstyle.INK, lw=1.4, zorder=5)


def direction_panel(ax, ys, G, xlim):
    """(b) zero-split violins of per-family signed grouped SHAP."""
    for y, col in zip(ys, range(G.shape[1])):
        _violin(ax, G[:, col], y, xlim)

    ax.axvline(0.0, color=figstyle.INK, lw=0.8, ls=(0, (4, 3)), zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, G.shape[1] - 0.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Grouped SHAP value  (margin)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.text(0.0, 1.012, "(b)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=figstyle.INK)

    # directional cue flanking the axis it describes — which end favors which mode
    ax.text(0.0, 0.012, "← favors non-abrupt", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.8, color=COOL)
    ax.text(1.0, 0.012, "favors abrupt →", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color=WARM)


def main():
    figstyle.use()
    labels, importance, G = load_cache()
    n = len(labels)
    ys = np.arange(n)[::-1]      # most important family at the top

    # Robust x-window for (b): cover every family's 2-98 pct, padded, 0 always in view.
    p = np.percentile(G, [2, 98], axis=0)
    lo, hi = float(p[0].min()), float(p[1].max())
    pad = 0.06 * (hi - lo)
    xlim = (min(0.0, lo - pad), hi + pad)

    fig = figstyle.figure("full", height=8.1, subplots=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.85], wspace=0.04,
                          left=0.235, right=0.985, top=0.95, bottom=0.065)
    ax_mag = fig.add_subplot(gs[0, 0])
    ax_dir = fig.add_subplot(gs[0, 1])

    magnitude_panel(ax_mag, ys, labels, importance)
    direction_panel(ax_dir, ys, G, xlim)

    figstyle.save(fig, "07_shap_families")
    plt.close(fig)
    print(f"wrote output/07_shap_families.{{pdf,png}}  ({n} families, N={G.shape[0]:,} points)")


if __name__ == "__main__":
    main()
