"""Figure 1 — thaw-mode concept schematic (panel a) + two abrupt-thaw field photos.

Panel (a) is the hand-built conceptual schematic (gate -> fork -> stage), kept as
the vector source `fig01_schematic_panel.pdf` and rasterized here at 600 dpi so
the composite is a single self-contained asset. Panels (b) and (c) are field
photographs of the two abrupt-thaw landforms the schematic names:
  (b) a retrogressive thaw slump / thermoerosional riverbank exposing massive
      ground ice  (thaw-slump-overeem.png)
  (c) thermokarst lakes in a boreal lowland            (thermokarst-lake-rozmiarek.jpg)

Photos carry NO on-image annotation by design; the vocabulary lives in the
schematic directly above and in the caption. Layout is absolute-positioned
(add_axes) with tight=False, per figstyle.save's mixed-mode raster guidance.

Run:  poetry run python output/fig01_process_schematic.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PHOTOS = REPO / "data" / "photos"
SCHEMATIC_PDF = HERE / "fig01_schematic_panel.pdf"
SLUMP = PHOTOS / "thaw-slump-overeem.png"
LAKE = PHOTOS / "thermokarst-lake-rozmiarek.jpg"

SCHEMATIC_DPI = 600  # fine-detail raster for the line-art + label panel
SCH_INSET = 0.94     # schematic width as a fraction of the photo-row width (even side margins)

# Panel-label styling copied from the schematic's own baked-in "(a)": plain black
# parenthesized sans, sitting at the top-left ABOVE the artwork (not overlaid).
LABEL_FONTSIZE = 13
BORDER_COLOR = "#444444"
BORDER_LW = 0.8


def rasterize_pdf(pdf: Path, dpi: int) -> Path:
    """Render a single-page PDF to PNG with poppler's pdftocairo."""
    out_prefix = HERE / f"{pdf.stem}_{dpi}dpi"
    out_png = out_prefix.with_suffix(".png")
    subprocess.run(
        ["pdftocairo", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(out_prefix)],
        check=True,
    )
    return out_png


def trim_whitespace(img, *, thresh: float = 0.97, pad: int = 30):
    """Crop an (H, W, C) array to its non-white content bounding box, plus a pad.

    Removes the schematic's asymmetric canvas margins so the drawn content can be
    re-centered with even whitespace on both sides.
    """
    rgb = img[..., :3] if img.shape[2] >= 3 else img
    mask = rgb.mean(axis=2) < thresh
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    h, w = mask.shape
    x0, x1 = max(cols.min() - pad, 0), min(cols.max() + 1 + pad, w)
    y0, y1 = max(rows.min() - pad, 0), min(rows.max() + 1 + pad, h)
    return img[y0:y1, x0:x1]


def crop_to_aspect(img, target_wh: float, *, x_anchor: float = 0.5, y_anchor: float = 0.5):
    """Center-ish crop an (H, W, C) array to a target width/height ratio.

    x_anchor / y_anchor in [0, 1] bias which part is kept when cropping that axis
    (0.5 = centered). Only the over-long axis is cropped; the other is untouched.
    """
    h, w = img.shape[:2]
    cur = w / h
    if cur > target_wh:  # too wide -> crop columns
        new_w = int(round(h * target_wh))
        x0 = int(round((w - new_w) * x_anchor))
        return img[:, x0:x0 + new_w]
    if cur < target_wh:  # too tall -> crop rows
        new_h = int(round(w / target_wh))
        y0 = int(round((h - new_h) * y_anchor))
        return img[y0:y0 + new_h, :]
    return img


def main() -> None:
    figstyle.use()

    schematic = trim_whitespace(mpimg.imread(rasterize_pdf(SCHEMATIC_PDF, SCHEMATIC_DPI)))
    slump = mpimg.imread(str(SLUMP))
    lake = mpimg.imread(str(LAKE))

    # Common bottom-row aspect (3:2 landscape); crop each photo to it.
    row_wh = 3.0 / 2.0
    slump = crop_to_aspect(slump, row_wh, y_anchor=0.5)          # 4:3 -> trim top/bottom
    lake = crop_to_aspect(lake, row_wh, x_anchor=0.35)           # 16:9 -> keep the left lake

    sch_wh = schematic.shape[1] / schematic.shape[0]             # trimmed content aspect

    # --- absolute layout in inches -> figure fractions --------------------- #
    W = figstyle.WIDTHS_IN["full"]        # 6.69 in (double column)
    m = 0.05                              # outer margin
    gutter = 0.10                         # between the two photos
    vgap = 0.06                           # schematic -> (b)/(c) label row
    label_h = 0.20                        # room for (b)/(c) above the photos

    content_w = W - 2 * m
    photo_w = (content_w - gutter) / 2.0
    photo_h = photo_w / row_wh

    sch_w = SCH_INSET * content_w                     # inset so side whitespace is even
    sch_h = sch_w / sch_wh
    sch_x = m + (content_w - sch_w) / 2.0             # horizontally centered over the row
    sch_y = m + photo_h + label_h + vgap

    H = sch_y + sch_h + m

    fig = figstyle.figure("full", height=H, subplots=False)

    def add(x_in, y_in, w_in, h_in):
        ax = fig.add_axes([x_in / W, y_in / H, w_in / W, h_in / H])
        return ax

    def bordered_photo(x_in, img, letter):
        ax = add(x_in, m, photo_w, photo_h)
        ax.imshow(img, aspect="auto", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set(visible=True, color=BORDER_COLOR, linewidth=BORDER_LW)
        # (b)/(c) label above the image, top-left, in the plain style of (a)
        fig.text((x_in) / W, (m + photo_h + 0.03) / H, f"({letter})",
                 ha="left", va="bottom", fontsize=LABEL_FONTSIZE, color=figstyle.INK)
        return ax

    # schematic on top (panel a is baked into the artwork), trimmed + centered
    ax_a = add(sch_x, sch_y, sch_w, sch_h)
    ax_a.set_axis_off()
    ax_a.imshow(schematic, rasterized=True)

    # photo row: (b) slump left, (c) thermokarst lake right
    bordered_photo(m, slump, "b")
    bordered_photo(m + photo_w + gutter, lake, "c")

    figstyle.save(fig, "01_process_schematic", rasterized_dpi=SCHEMATIC_DPI, tight=False)
    print(f"wrote {HERE / '01_process_schematic.pdf'}  ({W:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
