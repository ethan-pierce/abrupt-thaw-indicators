"""Figure 4 — Spatial out-of-sample performance (L2a, L2c, L3).

The credibility figure. Its single load-bearing claim: the thaw-mode signal is
NOT a location proxy — it survives being forced to reach across space. Two panels,
each refuting a distinct form of the proxy objection, both scoring the SAME
operative model (`models/selected_hparams.json`, spw=1) so the figure reads as one
model under progressively harder spatial regimes:

  (a) block-size ladder — refutes SHORT-RANGE leakage. Repeated spatial block-CV
      (20 partition reshuffles/scale, `diagnostics/repeated_cv.py`): XGBoost holds
      as blocks grow 5->200 km, over a logistic floor, far above the prevalence
      floor. Error band = across-partition sigma (the honest uncertainty, not one
      partition's fold luck). Caveat (caption): hyperparameters held fixed here, so
      the per-fold selection cost is not re-paid.

  (b) leave-region-out extrapolation — refutes REGION MEMORIZATION. AUC-PR vs how
      far the model must reach (median distance from a held-out point to its nearest
      training point, `diagnostics/extrapolation_range.py`); graceful decay toward
      the floor as held-out regions coarsen from 50 to 3.

House rules: monochrome value ladder (INK hero / MUTED baseline / gray floor) — hue
is reserved for class semantics, never model identity. Floor is the only reference
anchor; no leaky-random-split ceiling. Numbers annotated on-figure are read live
from the cached JSON, so they can never drift from the plotted points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import figstyle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
REPCV_JSON = HERE / "repeated_cv_results.json"
EXTRAP_JSON = HERE / "extrapolation_range_results.json"

# Model-series colors from the house Okabe-Ito qualitative palette (the sanctioned
# choice for incidental, non-class categoricals). Deliberately NOT the warm/cool
# class poles — these encode model identity, not thaw class. XGBoost is the operative
# model shown in BOTH panels, so it keeps one color throughout to read as one model.
XGB_COLOR = figstyle.QUALITATIVE[4]     # blue   #0072B2
LOGIT_COLOR = figstyle.QUALITATIVE[0]   # orange #E69F00


def panel_a_blocksize(ax, d):
    """Repeated block-CV ladder: XGBoost (hero) over logistic over the chance floor."""
    scales = np.array(d["scales_km"], dtype=float)
    ps = d["per_scale"]
    xgb_m = np.array([ps[str(int(s))]["xgb_mean"] for s in scales])
    xgb_sd = np.array([ps[str(int(s))]["xgb_std"] for s in scales])
    logit_m = np.array([ps[str(int(s))]["logit_mean"] for s in scales])
    floor = d["prevalence_floor"]
    op_km = d["operative_cell_km"]

    # chance floor (only reference anchor)
    ax.axhline(floor, color=figstyle.OTHER_GRAY, linestyle=":", linewidth=1.0,
               zorder=1, label=f"prevalence floor ({floor:.3f})")
    # logistic baseline — orange, dashed, no markers (below the hero in hierarchy)
    ax.plot(scales, logit_m, color=LOGIT_COLOR, linestyle="--", linewidth=1.4,
            zorder=2, label="Logistic (baseline)")
    # XGBoost hero — blue, solid, markers, across-partition sigma band
    ax.fill_between(scales, xgb_m - xgb_sd, xgb_m + xgb_sd, color=XGB_COLOR,
                    alpha=0.18, linewidth=0, zorder=2.5)
    ax.plot(scales, xgb_m, color=XGB_COLOR, linestyle="-", linewidth=1.8,
            marker="o", markersize=4.5, zorder=3, label="XGBoost (operative)")

    # annotate the operative 10 km value, exact from JSON
    op_val = ps[str(int(op_km))]["xgb_mean"]
    ax.annotate(f"{op_val:.2f}", xy=(op_km, op_val), xytext=(op_km, op_val + 0.11),
                ha="center", va="bottom", fontsize=7.5, color=figstyle.INK,
                arrowprops=dict(arrowstyle="-", color=figstyle.INK, lw=0.6,
                                shrinkA=0, shrinkB=2))

    ax.set_xscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([str(int(s)) for s in scales])
    ax.tick_params(axis="x", which="minor", length=0)  # no minor log ticks between our scales
    ax.set_xlabel("Spatial block size (km)")
    ax.set_ylabel("AUC-PR")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.13), fontsize=6.8,
              frameon=False, handlelength=1.8, borderaxespad=0.0)
    figstyle.panel_label(ax, "a")


def panel_b_extrapolation(ax, d):
    """Leave-region-out decay: single INK curve vs distance, region-count labels."""
    rows = sorted(d["rows"], key=lambda r: r["med_dist_km"])  # ascending distance
    dist = np.array([r["med_dist_km"] for r in rows])
    ap = np.array([r["ap"] for r in rows])
    regions = [r["regions"] for r in rows]
    floor = d["prevalence_floor"]

    ax.axhline(floor, color=figstyle.OTHER_GRAY, linestyle=":", linewidth=1.0, zorder=1)
    ax.annotate(f"prevalence floor ({floor:.3f})", xy=(dist.max(), floor),
                xytext=(-2, 4), textcoords="offset points", ha="right", va="bottom",
                fontsize=6.8, color=figstyle.MUTED)

    ax.plot(dist, ap, color=XGB_COLOR, linestyle="-", linewidth=1.8,
            marker="o", markersize=4.5, zorder=3)

    # region-count labels, fanned up-right of each point (small, muted) so the
    # crowded fine-granularity cluster on the left doesn't overlap.
    for x, yv, g in zip(dist, ap, regions):
        ax.annotate(str(g), xy=(x, yv), xytext=(4, 5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=6.5, color=figstyle.MUTED)

    # annotate the farthest-reach endpoint value, exact from JSON
    xe, ye = dist[-1], ap[-1]
    ax.annotate(f"{ye:.2f}", xy=(xe, ye), xytext=(-6, -12), textcoords="offset points",
                ha="right", va="top", fontsize=7.5, color=figstyle.INK,
                arrowprops=dict(arrowstyle="-", color=figstyle.INK, lw=0.6,
                                shrinkA=0, shrinkB=2))

    ax.set_xlim(0, dist.max() * 1.06)
    ax.set_xlabel("Distance to nearest training point (km)")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="y", labelleft=False)   # shared y — labels on panel (a) only
    # name what the point labels mean, up in the empty top-right (clear of the floor note)
    ax.text(0.97, 0.95, "labels = # of held-out regions", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=figstyle.MUTED)
    figstyle.panel_label(ax, "b")


def main():
    figstyle.use()
    rep = json.loads(REPCV_JSON.read_text())
    ext = json.loads(EXTRAP_JSON.read_text())

    fig, (ax_a, ax_b) = figstyle.figure("full", height=2.9, ncols=2, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.145, wspace=0.06)

    panel_a_blocksize(ax_a, rep)
    panel_b_extrapolation(ax_b, ext)

    figstyle.save(fig, "04_spatial_performance")
    op = rep["per_scale"][str(int(rep["operative_cell_km"]))]["xgb_mean"]
    far = sorted(ext["rows"], key=lambda r: r["med_dist_km"])[-1]
    print(f"wrote 04_spatial_performance.pdf/.png  "
          f"(a: XGBoost {op:.3f} @ {rep['operative_cell_km']} km; "
          f"b: {far['ap']:.3f} @ {far['med_dist_km']:.0f} km / {far['regions']} regions)")


if __name__ == "__main__":
    main()
