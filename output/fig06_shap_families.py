"""Figure 6 — SHAP importance of the 22 emergent feature families (a), with the
Land Cover family decomposed by land-cover class (b)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import figstyle  # noqa: E402
import shap_family_display as fd  # noqa: E402

GROUPED = _HERE / "shap_grouped_matrix.npz"
MECHANISM = _HERE / "shap_mechanism_cache.npz"

LAND_COVER = "Land Cover (18 classes)"
ACCENT = "#4d4d4d"

BAR = 0.62
LINE = 0.34
MEMBER_OFFSET = 0.26
LABEL_PAD = 0.04
NAME_SIZE = 8
MEMBER_SIZE = 6.5
VALUE_SIZE = 7.2

CLASS_N_MIN = 100
CLASS_NAMES = {
    "Open Water": "Open water",
    "Emergent Herbaceous Wetlands": "Emergent wetlands",
    "Shrub/Scrub": "Shrub/scrub",
    "Dwarf Scrub": "Dwarf scrub",
    "Barren Land (Rock/Sand/Clay)": "Barren land",
    "Deciduous Forest": "Deciduous forest",
    "Woody Wetlands": "Woody wetlands",
    "Evergreen Forest": "Evergreen forest",
    "Sedge/Herbaceous": "Sedge/herbaceous",
}
CLASS_SIZE = 7.5


def load_families():
    d = np.load(GROUPED, allow_pickle=True)
    return list(d["labels"]), d["importance"].astype(float)


def load_land_cover_classes():
    d = np.load(MECHANISM, allow_pickle=True)
    names = list(d["feature_names"])
    values, data = d["values"].astype(float), d["data"].astype(float)
    cols = [c for c in names if c.startswith("Land Cover")]
    family_shap = values[:, [names.index(c) for c in cols]].sum(axis=1)
    rows = []
    for c in cols:
        present = data[:, names.index(c)] == 1
        if present.sum() >= CLASS_N_MIN:
            vals = family_shap[present]
            rows.append((c[len("Land Cover ("):-1], vals,
                         present.mean() * 100, float(np.median(vals))))
    return sorted(rows, key=lambda r: r[3])


def bar_color(label):
    return ACCENT if label == LAND_COVER else fd.color(label)


def text_width(fig, s):
    t = fig.text(0, 0, s, fontsize=MEMBER_SIZE)
    w = t.get_window_extent(fig.canvas.get_renderer()).width
    t.remove()
    return w


def wrap_members(fig, items, max_width):
    lines = [items[0]]
    for item in items[1:]:
        joined = f"{lines[-1]}, {item}"
        if text_width(fig, joined + ",") > max_width:
            lines[-1] += ","
            lines.append(item)
        else:
            lines[-1] = joined
    return lines


def family_panel(ax, labels, importance):
    fig = ax.figure
    share = importance / importance.sum() * 100
    box = ax.get_window_extent()
    max_width = box.x0 - LABEL_PAD * box.width - 6
    members = [wrap_members(fig, fd.MEMBERS[l], max_width) if l in fd.MEMBERS else []
               for l in labels]
    pitch = [1 + max(0.0, MEMBER_OFFSET + LINE * len(m) - BAR / 2) for m in members]
    ys = -np.concatenate([[0], np.cumsum(pitch[:-1])])
    label_x = ax.get_yaxis_transform()

    ax.barh(ys, share, height=BAR, color=[bar_color(l) for l in labels])
    for y, label, s, m in zip(ys, labels, share, members):
        ax.annotate(f"{s:.0f}%" if s >= 0.5 else "<1%", xy=(s, y), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=VALUE_SIZE,
                    color=figstyle.INK)
        ax.text(-LABEL_PAD, y, fd.NAMES[label], transform=label_x, ha="right",
                va="center", fontsize=NAME_SIZE, color=figstyle.INK,
                fontweight="bold" if label == LAND_COVER else "normal")
        for i, line in enumerate(m):
            ax.text(-LABEL_PAD, y - MEMBER_OFFSET - LINE * (i + 0.5), line,
                    transform=label_x, ha="right", va="center", fontsize=MEMBER_SIZE,
                    color=figstyle.MUTED)

    ax.set_xlim(0, share.max() * 1.12)
    ax.set_ylim(ys[-1] - pitch[-1] + 0.4, 0.5)
    ax.axis("off")
    return ys, share


def land_cover_panel(ax, rows):
    ypos = np.arange(len(rows))
    bp = ax.boxplot([r[1] for r in rows], vert=False, positions=ypos, widths=0.55,
                    whis=(5, 95), showfliers=False, patch_artist=True)
    for r, box, med in zip(rows, bp["boxes"], bp["medians"]):
        box.set_facecolor(figstyle.ABRUPT if r[3] >= 0 else figstyle.NON_ABRUPT)
        box.set_alpha(0.85)
        box.set_edgecolor(figstyle.INK)
        box.set_linewidth(0.6)
        med.set_color(figstyle.INK)
        med.set_linewidth(1.0)
    for part in ("whiskers", "caps"):
        for artist in bp[part]:
            artist.set_color(figstyle.INK)
            artist.set_linewidth(0.6)
    ax.axvline(0, color=figstyle.INK, lw=0.7, zorder=1)

    for (name, vals, share, _), y in zip(rows, ypos):
        lo, hi = np.percentile(vals, [5, 95])
        x, ha = (lo - 0.05, "right") if hi > 1.0 else (max(hi, 0) + 0.05, "left")
        ax.text(x, y, f"{CLASS_NAMES[name]} ({share:.0f}%)", ha=ha, va="center",
                fontsize=CLASS_SIZE, color=figstyle.INK)

    ax.set_xlim(-0.8, 1.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([-0.5, 0, 0.5, 1.0, 1.5])
    ax.tick_params(axis="x", labelsize=7, length=2.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Land cover SHAP (margin)", fontsize=7.5, labelpad=2)
    for x, ha, text, color in ((0.0, "left", "← Non-abrupt", figstyle.NON_ABRUPT),
                               (1.0, "right", "Abrupt →", figstyle.ABRUPT)):
        ax.text(x, -0.08, text, transform=ax.transAxes, ha=ha, va="top",
                fontsize=7.5, color=color, fontweight="bold")


def main():
    figstyle.use()
    labels, importance = load_families()

    fig = figstyle.figure("full", height=8.6, subplots=False)
    ax_a = fig.add_axes([0.42, 0.01, 0.56, 0.98])
    ys, share = family_panel(ax_a, labels, importance)

    ax_b = fig.add_axes([0.62, 0.20, 0.36, 0.44])
    land_cover_panel(ax_b, load_land_cover_classes())

    i = labels.index(LAND_COVER)
    fig.add_artist(ConnectionPatch(
        xyA=(share[i], ys[i]), coordsA=ax_a.transData,
        xyB=(0.5, 1.02), coordsB=ax_b.transAxes,
        color=ACCENT, lw=0.7, connectionstyle="angle,angleA=0,angleB=90",
        arrowstyle="-", shrinkA=18))

    fig.text(0.005, 0.995, "(a)", ha="left", va="top", fontsize=9, fontweight="bold",
             color=figstyle.INK)
    ax_b.text(-0.02, 1.05, "(b)", transform=ax_b.transAxes, ha="left", va="bottom",
              fontsize=9, fontweight="bold", color=figstyle.INK)

    figstyle.save(fig, "06_shap_families")
    plt.close(fig)
    print(f"wrote output/06_shap_families.{{pdf,png}}  ({len(labels)} families)")


if __name__ == "__main__":
    main()
