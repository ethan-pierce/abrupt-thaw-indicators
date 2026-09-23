"""Figure 6 — SHAP importance of the 22 emergent feature families."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import figstyle  # noqa: E402
import shap_family_display as fd  # noqa: E402

GROUPED = _HERE / "shap_grouped_matrix.npz"

BAR = 0.62
LINE = 0.3
MEMBER_OFFSET = 0.22
LABEL_PAD = 0.04
MEMBER_SIZE = 5.6


def load_families():
    d = np.load(GROUPED, allow_pickle=True)
    return list(d["labels"]), d["importance"].astype(float)


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

    ax.barh(ys, share, height=BAR, color=[fd.color(l) for l in labels])
    for y, label, s, m in zip(ys, labels, share, members):
        ax.annotate(f"{s:.0f}%" if s >= 0.5 else "<1%", xy=(s, y), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.8,
                    color=figstyle.INK)
        ax.text(-LABEL_PAD, y, fd.NAMES[label], transform=label_x, ha="right",
                va="center", fontsize=7.3, color=figstyle.INK)
        for i, line in enumerate(m):
            ax.text(-LABEL_PAD, y - MEMBER_OFFSET - LINE * (i + 0.5), line,
                    transform=label_x, ha="right", va="center", fontsize=MEMBER_SIZE,
                    color=figstyle.MUTED)

    ax.set_xlim(0, share.max() * 1.12)
    ax.set_ylim(ys[-1] - pitch[-1] + 0.4, 0.5)
    ax.axis("off")


def main():
    figstyle.use()
    labels, importance = load_families()

    fig = figstyle.figure("onehalf", height=7.7, subplots=False)
    ax = fig.add_axes([0.47, 0.01, 0.51, 0.98])
    family_panel(ax, labels, importance)

    figstyle.save(fig, "06_shap_families")
    plt.close(fig)
    print(f"wrote output/06_shap_families.{{pdf,png}}  ({len(labels)} families)")


if __name__ == "__main__":
    main()
