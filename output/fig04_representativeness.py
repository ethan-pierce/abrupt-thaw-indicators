"""Figure 4 — Representativeness of the training sample.

Compares, per feature, the marginal distribution of the training sample against
the in-AOA statewide grid the model scores. The training sample is lake- and
road-biased, so it sits in flatter, lower, wetter locations than the grid. The
AOA measures coverage, not density: a sample can span every covariate's range
while over-representing part of it, and this figure shows that density shift.

Train side: features_clean.csv via diagnostics/_data.load.
Grid side: the full in-AOA distribution from prediction_data.nc masked by
aoa.nc (DI <= threshold).

Writes output/04_representativeness.{pdf,png}.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "diagnostics"))

import figstyle  # noqa: E402
from settings import DATA  # noqa: E402
import _data  # noqa: E402

FILL = -9999.0
GRID_SUBSAMPLE = 150_000
RNG = np.random.default_rng(42)

# Ridge geometry: grid ridge at the axis, training ridge raised to clear it.
GRID_BASE = 0.0
TRAIN_BASE = 1.15
RIDGE_H = 0.9


def sig2(v, n=2):
    """v to n significant figures in fixed (non-scientific) notation.

    Two significant figures matches the sample's precision: a fraction near
    0.08 from ~19k points has a standard error of ~0.002.
    """
    from math import floor, log10
    if not np.isfinite(v) or v == 0:
        return "0"
    d = max(0, n - 1 - int(floor(log10(abs(v)))))
    return f"{v:.{d}f}"

TRAIN_COLOR = "#009E73"   # green — the training sample (under scrutiny)
GRID_COLOR = "#CC79A7"    # reddish-purple — the in-AOA statewide grid (reference)

# Continuous features: (column, axis label, arcsinh linear-threshold, raw x-ticks).
# arcsinh(v / linthresh) is linear near 0 (zeros map to 0 exactly) and log-like
# in the heavy tail.
CONTINUOUS = [
    ("Slope", "Slope (°)", 1.0, [0, 1, 3, 10, 30]),
    ("Height Above Nearest Drainage", "Height above nearest drainage (m)", 1.0,
     [0, 1, 10, 100, 1000]),
    ("Elevation", "Elevation (m)", 10.0, [0, 10, 100, 1000]),
    ("Upstream Area", "Upstream area (km²)", 0.01, [0, 0.1, 10]),
]

# One-hot features: (column, short label).
ONEHOT = [
    ("Land Cover (Open Water)", "Open water"),
    ("Land Cover (Emergent Herbaceous Wetlands)", "Emergent herbaceous wetland"),
    ("Land Cover (Deciduous Forest)", "Deciduous forest"),
]


def load_data():
    """Return (X_train, in_aoa_grid_values_by_feature)."""
    X, _y, _lat, _lon = _data.load(verify=True)

    sus = xr.open_dataset(DATA / "susceptibility.nc")
    aoa = xr.open_dataset(DATA / "aoa.nc")
    cube = xr.open_dataset(DATA / "prediction_data.nc")

    le = sus["log_evidence"].values
    di = aoa["DI"].values
    thr = float(aoa.attrs["aoa_threshold"])
    in_aoa = np.isfinite(le) & np.isfinite(di) & (di <= thr)
    print(f"in-AOA grid cells: {int(in_aoa.sum()):,} (threshold DI <= {thr:.4f})")

    feats = cube["feature"].values.tolist()
    wanted = [c for c, *_ in CONTINUOUS] + [c for c, *_ in ONEHOT]
    grid = {}
    for name in wanted:
        slab = cube["feature_stack"].isel(feature=feats.index(name)).values
        slab = np.where(slab == FILL, np.nan, slab)
        vals = slab[in_aoa]
        grid[name] = vals[np.isfinite(vals)]
    sus.close(); aoa.close(); cube.close()
    return X, grid


def _asinh(v, lt):
    return np.arcsinh(np.asarray(v, float) / lt)


def _ridge(ax, t, baseline, color, *, span, label=None):
    """Draw one filled KDE ridge from `baseline`; return the peak height used."""
    xs = np.linspace(span[0], span[1], 400)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        kde = gaussian_kde(t)
        d = kde(xs)
    d = d / d.max() * RIDGE_H  # normalize each ridge to a common visual height
    ax.fill_between(xs, baseline, baseline + d, color=color, alpha=0.55,
                    lw=0.0, zorder=2)
    ax.plot(xs, baseline + d, color=color, lw=1.1, zorder=3, label=label)
    return d


def _median_mark(ax, raw_median, lt, baseline, color):
    """Vertical median tick within a ridge band + numeric label above it."""
    t = _asinh(raw_median, lt)
    ax.plot([t, t], [baseline, baseline + RIDGE_H], color=color, lw=1.3, zorder=4)
    ax.annotate(sig2(raw_median), xy=(t, baseline + RIDGE_H), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.5, color=figstyle.INK, zorder=5)


def continuous_panel(ax, xlabel, lt, ticks, train_vals, grid_vals, letter):
    tv = np.asarray(train_vals, float); tv = tv[np.isfinite(tv)]
    gv = np.asarray(grid_vals, float); gv = gv[np.isfinite(gv)]

    grid_median = float(np.median(gv))
    train_median = float(np.median(tv))

    # Subsample only for the KDE; reported medians use the full population.
    gv_kde = gv
    if gv_kde.size > GRID_SUBSAMPLE:
        gv_kde = RNG.choice(gv_kde, GRID_SUBSAMPLE, replace=False)

    t_train = _asinh(tv, lt)
    t_grid = _asinh(gv_kde, lt)
    lo = min(t_train.min(), t_grid.min())
    hi = max(t_train.max(), t_grid.max())
    pad = 0.05 * (hi - lo)
    span = (lo - pad, hi + pad)

    # grid ridge at the axis, train ridge raised clear of it — a two-group ridgeline.
    _ridge(ax, t_grid, GRID_BASE, GRID_COLOR, span=span)
    _ridge(ax, t_train, TRAIN_BASE, TRAIN_COLOR, span=span)
    _median_mark(ax, grid_median, lt, GRID_BASE, GRID_COLOR)
    _median_mark(ax, train_median, lt, TRAIN_BASE, TRAIN_COLOR)

    ax.set_xlim(*span)
    ax.set_ylim(-0.05, TRAIN_BASE + RIDGE_H + 0.2)
    ax.set_yticks([])
    ticks = [v for v in ticks if span[0] <= _asinh(v, lt) <= span[1]]
    ax.set_xticks([_asinh(v, lt) for v in ticks])
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    # Letter above the axes so it clears the train-ridge median labels at top-left.
    ax.text(0.0, 1.05, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=9, fontweight="bold", color=figstyle.INK)


def onehot_panel(ax, name, train_frac, grid_frac, letter):
    """One land-cover class: a train-vs-grid paired horizontal bar."""
    h = 0.34
    ax.barh(h / 2 + 0.03, train_frac, height=h, color=TRAIN_COLOR, alpha=0.9, zorder=2)
    ax.barh(-h / 2 - 0.03, grid_frac, height=h, color=GRID_COLOR, alpha=0.9, zorder=2)
    ax.annotate(sig2(train_frac), xy=(train_frac, h / 2 + 0.03), xytext=(3, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=7, color=figstyle.INK)
    ax.annotate(sig2(grid_frac), xy=(grid_frac, -h / 2 - 0.03), xytext=(3, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=7, color=figstyle.INK)
    ax.text(0.5, 1.0, name, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color=figstyle.INK)
    ax.set_yticks([])
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlim(0, max(train_frac, grid_frac) * 1.22)
    ax.set_xlabel("Fraction of locations", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    figstyle.panel_label(ax, letter, loc="upper right")


def main():
    figstyle.use()
    figstyle.assert_cvd_safe([TRAIN_COLOR, GRID_COLOR], name="train/grid pair")
    X, grid = load_data()

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = figstyle.figure("full", height=4.8, subplots=False)
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1.2, 0.8], hspace=0.62, wspace=0.6,
        left=0.03, right=0.985, top=0.86, bottom=0.11,
    )

    # Top row: 4 continuous ridgelines, 3 grid-columns each.
    letters = iter("abcdefg")
    for i, (col, xlabel, lt, ticks) in enumerate(CONTINUOUS):
        ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3])
        continuous_panel(ax, xlabel, lt, ticks,
                         X[col].values, grid[col], next(letters))

    # Bottom row: 3 one-hot paired-bar panels, 4 grid-columns each.
    train_fracs = [float(np.nanmean(X[col].values)) for col, _ in ONEHOT]
    grid_fracs = [float(np.nanmean(grid[col])) for col, _ in ONEHOT]
    for j, (col, lab) in enumerate(ONEHOT):
        ax = fig.add_subplot(gs[1, j * 4:(j + 1) * 4])
        onehot_panel(ax, lab, train_fracs[j], grid_fracs[j], next(letters))

    handles = [Patch(facecolor=TRAIN_COLOR, alpha=0.7, label="Training sample"),
               Patch(facecolor=GRID_COLOR, alpha=0.7, label="In-AOA statewide grid")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.985))

    figstyle.save(fig, "04_representativeness")
    plt.close(fig)
    print("wrote output/04_representativeness.{pdf,png}")


if __name__ == "__main__":
    main()
