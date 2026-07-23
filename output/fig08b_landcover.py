"""Figure 8b — SHAP mechanism: Land Cover per class (L6b, categorical analog).

Fig 8 shows the CONTINUOUS dependence shapes (own SHAP vs own value) for the top-9
interpretable features. Land Cover is the one important family a dependence plot
cannot handle — a one-hot has no continuous axis — so it is split off here and
DECOMPOSED per class: one horizontal box (median, IQR, 5-95 whiskers) of the Land
Cover *family* SHAP among the training points of that class.

Land Cover is also the one family that discriminates in BOTH directions (its family
SHAP straddles zero): Open Water strongly favors Abrupt (lake/lowland thermokarst —
and the most sampling-biased class, ~43% of points), while Sedge/Herbaceous and the
forest/scrub classes favor Non-abrupt.

Framing discipline (locked, matches Fig 8): shapes reported as FACT about the
model's response, never asserted mechanism; §5.2 carries interpretation.

Data: output/shap_mechanism_cache.npz — per-feature OOF SHAP (Abrupt-oriented:
positive => favors Abrupt) + feature VALUES + names, written by
models/shap_mechanism_cache.py. The family SHAP formed here (sum over the 18 Land
Cover one-hot columns) equals the Land Cover column of shap_grouped_matrix.npz by
construction. Pure plotting. Writes output/08b_landcover_shap.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import figstyle  # noqa: E402

CACHE = _HERE / "shap_mechanism_cache.npz"

WARM = figstyle.ABRUPT          # class median SHAP > 0  -> favors Abrupt
COOL = figstyle.NON_ABRUPT      # class median SHAP < 0  -> favors Non-abrupt
CLASS_N_MIN = 100               # drop land-cover classes below this many points


def load():
    d = np.load(CACHE, allow_pickle=True)
    names = list(d["feature_names"])
    return names, d["values"].astype(float), d["data"].astype(float)


def main():
    figstyle.use()
    names, values, data = load()

    lc_cols = [c for c in names if c.startswith("Land Cover")]
    idx = [names.index(c) for c in lc_cols]
    g = values[:, idx].sum(axis=1)                      # per-point Land Cover family SHAP
    n_total = data.shape[0]

    rows, om_n = [], 0
    for c in lc_cols:
        present = data[:, names.index(c)] == 1
        n = int(present.sum())
        if n >= CLASS_N_MIN:
            vals = g[present]
            rows.append((c[len("Land Cover ("):-1], vals, n,     # strip only the outer "Land Cover ( … )"
                         float(np.median(vals))))
        else:
            om_n += n
    rows.sort(key=lambda r: r[3])                       # ascending median: cool bottom, warm top
    om_classes = len(lc_cols) - len(rows)
    om_share = om_n / n_total * 100

    labels = [r[0] for r in rows]
    box_data = [r[1] for r in rows]
    ypos = np.arange(len(rows))

    w = figstyle.WIDTHS_IN["onehalf"]
    fig, ax = plt.subplots(figsize=(w, 4.0))
    fig.subplots_adjust(left=0.36, right=0.90, top=0.90, bottom=0.185)

    bp = ax.boxplot(box_data, vert=False, positions=ypos, widths=0.62,
                    whis=(5, 95), showfliers=False, patch_artist=True)
    for r, box, med in zip(rows, bp["boxes"], bp["medians"]):
        box.set_facecolor(WARM if r[3] >= 0 else COOL)
        box.set_alpha(0.85)
        box.set_edgecolor(figstyle.INK)
        box.set_linewidth(0.8)
        med.set_color(figstyle.INK)
        med.set_linewidth(1.3)
    for part in ("whiskers", "caps"):
        for artist in bp[part]:
            artist.set_color(figstyle.INK)
            artist.set_linewidth(0.8)

    ax.axvline(0, color=figstyle.INK, lw=0.9, zorder=1)

    # x-range from the whisker extremes, with right room for the share column
    lo = min(np.percentile(r[1], 5) for r in rows)
    hi = max(np.percentile(r[1], 95) for r in rows)
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)

    # share-of-points column at the right edge
    for r, y in zip(rows, ypos):
        share = r[2] / n_total * 100
        ax.annotate(f"{share:.0f}%" if share >= 0.5 else "<1%",
                    xy=(1.0, y), xytext=(6, 0), xycoords=("axes fraction", "data"),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=7.5, color=figstyle.MUTED)
    ax.annotate("share of\npoints", xy=(1.0, len(rows) - 0.3), xytext=(6, 4),
                xycoords="axes fraction", textcoords="offset points",
                va="bottom", ha="left", fontsize=7, color=figstyle.MUTED,
                linespacing=0.95)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.set_xlabel("SHAP  (margin)", fontsize=9.5)
    # direction on the x-axis ends (color + position carry "favors"), matching Fig 8
    ax.text(0.0, -0.175, "Non-abrupt", transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=COOL, fontweight="bold")
    ax.text(1.0, -0.175, "Abrupt", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=WARM, fontweight="bold")

    ax.set_title("Land Cover family SHAP, by class", fontsize=11, color=figstyle.INK,
                 pad=6)

    figstyle.save(fig, "08b_landcover_shap", outdir=_HERE, tight=False)
    print(f"Wrote 08b_landcover_shap.{{pdf,png}} | {len(rows)} classes (n≥{CLASS_N_MIN}); "
          f"{om_classes} omitted ({om_share:.1f}% of points)")
    for lab, vals, n, med in reversed(rows):
        print(f"   {lab:34s} n={n:5d} ({n / n_total * 100:4.1f}%)  median SHAP={med:+.2f}")


if __name__ == "__main__":
    main()
