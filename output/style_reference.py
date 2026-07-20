"""Render the house-style reference card — a living demo of `figstyle`.

Not a manuscript figure; it documents the palette, colormap, typography, and
panel-label conventions in one glance. Regenerate whenever the style changes:

    poetry run python style_reference.py
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np

import figstyle as fs


def main() -> None:
    fs.use()
    fig = fs.figure("full", height=3.4, subplots=False)

    # (a) class + support swatches -----------------------------------------
    axa = fig.add_axes([0.04, 0.12, 0.28, 0.74])
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    axa.axis("off")
    fs.panel_label(axa, "a")
    swatches = [
        (fs.CLASS_COLORS[0], "Abrupt (0) — warm", "#ffffff"),
        (fs.CLASS_COLORS[1], "Non-abrupt (1) — cool", "#ffffff"),
        (fs.DOMAIN_GRAY, "Permafrost domain", fs.INK),
        (fs.OTHER_GRAY, 'Family "Other"', fs.INK),
    ]
    y = 0.80
    for color, label, txt in swatches:
        axa.add_patch(matplotlib.patches.Rectangle((0.02, y - 0.06), 0.16, 0.11,
                                                    facecolor=color, edgecolor="none"))
        axa.text(0.06, y, label[:0], color=txt)  # keep swatch clean
        axa.text(0.22, y - 0.005, label, va="center", fontsize=8, color=fs.INK)
        y -= 0.16
    # out-of-AOA = hatch, not a value color
    axa.add_patch(matplotlib.patches.Rectangle((0.02, y - 0.06), 0.16, 0.11,
                                               facecolor="none", edgecolor=fs.MASK_COLOR,
                                               hatch=fs.MASK_HATCH, linewidth=0.0))
    axa.text(0.22, y - 0.005, "Out-of-AOA (hatch)", va="center", fontsize=8, color=fs.INK)

    # (b) log-evidence diverging scale, symmetric about 0 -------------------
    axb = fig.add_axes([0.40, 0.30, 0.30, 0.50])
    grad = np.linspace(-3, 3, 256).reshape(1, -1)
    im = axb.imshow(grad, aspect="auto", cmap=fs.LOG_EVIDENCE_CMAP,
                    norm=fs.symmetric_norm(3.0),
                    extent=[-3, 3, 0, 1], rasterized=True)
    axb.set_yticks([])
    axb.set_xticks([-3, 0, 3])
    axb.set_xlabel("log-evidence  (0 = neutral)")
    axb.text(-3, 1.15, "favors non-abrupt", fontsize=6.5, color=fs.MUTED, ha="left")
    axb.text(3, 1.15, "favors abrupt", fontsize=6.5, color=fs.MUTED, ha="right")
    fs.panel_label(axb, "b", loc="upper left")

    # (c) qualitative (Okabe-Ito) + typography -----------------------------
    axc = fig.add_axes([0.74, 0.12, 0.23, 0.74])
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")
    fs.panel_label(axc, "c")
    for i, c in enumerate(fs.QUALITATIVE):
        axc.add_patch(matplotlib.patches.Rectangle((i / len(fs.QUALITATIVE), 0.80),
                                                    1 / len(fs.QUALITATIVE), 0.10,
                                                    facecolor=c, edgecolor="none"))
    axc.text(0, 0.72, "Qualitative (Okabe-Ito), CVD-safe", fontsize=7, color=fs.MUTED)
    axc.text(0, 0.55, "Source Sans 3", fontsize=13, fontweight="bold", color=fs.INK)
    axc.text(0, 0.42, "axis label 8 pt", fontsize=8, color=fs.INK)
    axc.text(0, 0.32, "tick / legend 7 pt", fontsize=7, color=fs.INK)
    axc.text(0, 0.23, "annotation 6.5 pt", fontsize=6.5, color=fs.MUTED)
    axc.text(0, 0.10, "(a) panel letter 9 pt bold", fontsize=9, fontweight="bold",
             color=fs.INK)

    fs.save(fig, "style_reference")


if __name__ == "__main__":
    main()
