"""Appendix figure: model skill (AUC-PR) versus feature-space dissimilarity.

Reads the per-bin skill table cached by diagnostics/aoa_calibration.py
(output/aoa_calibration_bins.json) and renders it in house style. Shows that
per-bin AUC-PR does not decay as the dissimilarity index rises, and stays far
above the (drifting) per-bin prevalence floor across the whole sampled range.
The operative rank-CDF bins are drawn with their DI span, so the sample is seen
to reach across and past the AoA threshold; the raw-z coordinate is overlaid as
a robustness check. The threshold is the feature-space envelope, not a skill
limit.

Run: poetry run python output/fig_aoa_skill_vs_di.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import figstyle

HERE = Path(__file__).resolve().parent
BINS_JSON = HERE / "aoa_calibration_bins.json"


def _clean(rows):
    rows = [r for r in rows if np.isfinite(r["ap"]) and np.isfinite(r["mid"])]
    return (np.array([r["mid"] for r in rows]),
            np.array([r["ap"] for r in rows]),
            np.array([r["prev"] for r in rows]),
            np.array([r["lo"] for r in rows]),
            np.array([r["hi"] for r in rows]))


def main():
    d = json.loads(BINS_JSON.read_text())
    mid, ap, prev, lo, hi = _clean(d["bins_rank"])
    zmid, zap, _, _, _ = _clean(d["bins_raw_z"])

    c_rank = figstyle.QUALITATIVE[2]   # green   — operative rank-CDF coordinate
    c_rawz = figstyle.QUALITATIVE[0]   # orange  — raw-z robustness check

    figstyle.use()
    fig, ax = figstyle.figure("onehalf", aspect=0.62)

    # Per-bin chance floor as a shaded region from 0 up to each bin's prevalence.
    ax.fill_between(mid, 0.0, prev, color=figstyle.DOMAIN_GRAY, zorder=1,
                    label="chance floor (per-bin prevalence)")
    ax.plot(mid, prev, "-", color=figstyle.MUTED, lw=1.0, zorder=2)

    # Raw-z coordinate (robustness check), secondary.
    ax.plot(zmid, zap, "s--", color=c_rawz, lw=1.3, ms=4, zorder=4,
            label="AUC-PR (raw-z DI)")

    # Operative rank-CDF coordinate.
    ax.plot(mid, ap, "o-", color=c_rank, lw=1.8, ms=5, zorder=6,
            label="AUC-PR (rank-CDF DI)")

    ax.set_xlim(0.0, max(mid.max(), zmid.max()) * 1.06)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Dissimilarity index")
    ax.set_ylabel("AUC-PR (positive class: non-abrupt)")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.40), frameon=False)

    figstyle.save(fig, "aoa_skill_vs_di")
    print("Wrote output/aoa_skill_vs_di.pdf and .png")


if __name__ == "__main__":
    main()
