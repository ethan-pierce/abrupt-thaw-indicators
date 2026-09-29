"""Figure 5 — Spatial out-of-sample performance.

  (a) Pooled out-of-fold precision-recall curve at the operative block scale:
      XGBoost over the logistic baseline over the chance floor. Band = +/-1 sigma
      across partition reshuffles.

  (b) AUC-PR vs median distance to the nearest training point, for both
      spatial-holdout geometries on one axis: block-CV (square tiles) and
      leave-region-out (contiguous clusters). This panel carries the
      spatial-heterogeneity uncertainty, so (a) does not repeat it.

The two (b) series are the same model under two geometries, split by marker
and line style. All annotated numbers are read from output/fig05_cache.npz.

Rebuild the cache first: poetry run python output/fig05_cache_build.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import figstyle

HERE = Path(__file__).resolve().parent
CACHE = HERE / "fig05_cache.npz"

XGB_COLOR = figstyle.QUALITATIVE[4]     # blue   #0072B2 — the operative model
LOGIT_COLOR = figstyle.QUALITATIVE[0]   # orange #E69F00 — the linear baseline
# Panel (b) encodes holdout GEOMETRY (both series are the same operative XGBoost),
# so it needs two hues distinct from each other AND from (a)'s blue/orange/yellow.
BLOCK_COLOR = figstyle.QUALITATIVE[2]   # bluish green #009E73
REGION_COLOR = figstyle.QUALITATIVE[6]  # reddish purple #CC79A7


def panel_a_prcurve(ax, d):
    """Pooled-OOF PR curve: XGBoost + across-partition band, logistic, floor."""
    rec = d["recall_grid"]
    floor = float(d["prevalence"])
    xm, xs = d["xgb_prec_mean"], d["xgb_prec_std"]
    lm = d["logit_prec_mean"]

    # no-skill PR baseline is horizontal at the positive prevalence
    ax.axhline(floor, color=figstyle.OTHER_GRAY, linestyle=":", linewidth=1.0,
               zorder=1, label=f"chance floor ({floor:.3f})")
    # logistic baseline (mean curve only — the hero carries the band)
    ax.plot(rec, lm, color=LOGIT_COLOR, linestyle="--", linewidth=1.4, zorder=2,
            label="Logistic")
    # XGBoost hero + across-partition sigma band
    ax.fill_between(rec, xm - xs, xm + xs, color=XGB_COLOR, alpha=0.18,
                    linewidth=0, zorder=2.5)
    ax.plot(rec, xm, color=XGB_COLOR, linestyle="-", linewidth=1.8, zorder=3,
            label="XGBoost")

    xap, xsd = float(d["xgb_ap_mean"]), float(d["xgb_ap_std"])
    lap, lsd = float(d["logit_ap_mean"]), float(d["logit_ap_std"])
    ax.text(0.05, 0.34, f"XGBoost AUC-PR\n{xap:.2f} $\\pm$ {xsd:.2f}",
            transform=ax.transAxes, fontsize=7.5, color=XGB_COLOR, va="top")
    ax.text(0.05, 0.16, f"Logistic AUC-PR\n{lap:.2f} $\\pm$ {lsd:.2f}",
            transform=ax.transAxes, fontsize=7.5, color=LOGIT_COLOR, va="top")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=6.8, frameon=False, handlelength=1.8,
              borderaxespad=0.4)
    figstyle.panel_label(ax, "a")


def panel_b_distance(ax, d):
    """Skill vs distance-to-training: block-CV and region-out on one axis, two hues."""
    bd, ba = d["block_dist"], d["block_ap"]
    bq25, bq75 = d["block_q25"], d["block_q75"]
    rd, ra = d["region_dist"], d["region_ap"]
    rq25, rq75 = d["region_q25"], d["region_q75"]

    # Each marker is a configuration's median nearest-train distance; the strip is its
    # 25-75th percentile, the spread its pooled AUC-PR is computed over.
    for x0, y0, lo, hi in zip(bd, ba, bq25, bq75):
        ax.plot([lo, hi], [y0, y0], color=BLOCK_COLOR, linewidth=3.4, alpha=0.35,
                solid_capstyle="round", zorder=2)
    for x0, y0, lo, hi in zip(rd, ra, rq25, rq75):
        ax.plot([lo, hi], [y0, y0], color=REGION_COLOR, linewidth=3.4, alpha=0.35,
                solid_capstyle="round", zorder=2)

    ax.plot(bd, ba, color=BLOCK_COLOR, linestyle="-", linewidth=1.7, marker="o",
            markersize=4.5, zorder=3, label="Block-CV")
    ax.plot(rd, ra, color=REGION_COLOR, linestyle="--", linewidth=1.7, marker="s",
            markersize=4.2, markerfacecolor="white", markeredgecolor=REGION_COLOR,
            markeredgewidth=1.3, zorder=4, label="Leave-region-out")

    # Label the nearest block-CV and farthest region-out points.
    i_near = int(np.argmin(bd))
    ax.annotate(f"{ba[i_near]:.2f}", xy=(bd[i_near], ba[i_near]), xytext=(2, 7),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=7.0, color=BLOCK_COLOR)
    i_far = int(np.argmax(rd))
    ax.annotate(f"{ra[i_far]:.2f}", xy=(rd[i_far], ra[i_far]), xytext=(-4, -8),
                textcoords="offset points", ha="right", va="top",
                fontsize=7.0, color=REGION_COLOR)

    ax.set_xscale("log")
    ax.set_xlim(0.9, 360)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Distance to nearest training point (km)")
    ax.set_ylabel("AUC-PR")
    ax.legend(loc="upper right", fontsize=6.8, frameon=False, handlelength=1.8,
              borderaxespad=0.4)
    figstyle.panel_label(ax, "b")


def main():
    figstyle.use()
    d = dict(np.load(CACHE, allow_pickle=False))

    fig, (ax_a, ax_b) = figstyle.figure("full", height=2.9, ncols=2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.145, wspace=0.20)

    panel_a_prcurve(ax_a, d)
    panel_b_distance(ax_b, d)

    figstyle.save(fig, "05_spatial_performance")
    print(f"wrote 05_spatial_performance.pdf/.png  "
          f"(a: XGBoost AUC-PR {float(d['xgb_ap_mean']):.3f} ± {float(d['xgb_ap_std']):.3f} "
          f"@ {int(d['op_km'])} km; b: {float(d['region_ap'][np.argmax(d['region_dist'])]):.3f} "
          f"@ {float(d['region_dist'].max()):.0f} km)")


if __name__ == "__main__":
    main()
