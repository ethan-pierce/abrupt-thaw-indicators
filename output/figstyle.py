"""House figure style for abrupt-thaw-indicators (Earth's Future / AGU, Wiley).

Import this in every figure script so the whole paper reads as one system:

    import figstyle

    figstyle.use()                                  # apply style + register fonts
    fig, ax = figstyle.figure("single", aspect=0.8) # canvas at an AGU column width
    ax.scatter(..., color=figstyle.CLASS_COLORS[0]) # Abrupt = warm
    figstyle.panel_label(ax, "a")
    figstyle.save(fig, "04_susceptibility_map")     # writes .pdf (canonical) + .png

The prose rules (what code can't enforce — no in-figure titles, provenance in
caption, mandatory map furniture, log-evidence language) live in STYLE.md.

Design decisions are recorded in STYLE.md; the load-bearing ones enforced here:
  * Continuous fields use Crameri Scientific Colour Maps (CVD-safe, uniform).
  * Log-evidence uses `vik`, normalized SYMMETRICALLY about 0 (pale = neutral).
  * Class colors are drawn from vik's poles: warm = Abrupt, cool = Non-abrupt.
  * PDF is canonical (vector, embedded TrueType fonts); a 300-dpi PNG rides along.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as _cmc
from matplotlib import font_manager as _fm
from matplotlib.colors import Normalize, to_hex, to_rgb

_HERE = Path(__file__).resolve().parent
_STYLE = _HERE / "abrupt_thaw.mplstyle"
_FONTS = _HERE / "fonts"

# --------------------------------------------------------------------------- #
# AGU / Wiley canvas geometry.
# Verified against AGU "Text & Graphics Requirements" and the Earth's Future
# (Wiley) graphics page: single column 50-85 mm, two-column 105-170 mm, figure
# height <= 228 mm, raster 300-600 ppi. We design at the upper column widths.
# --------------------------------------------------------------------------- #
_MM = 1.0 / 25.4
WIDTHS_IN = {
    "single": 85.0 * _MM,   # 3.35 in  — one column
    "onehalf": 140.0 * _MM,  # 5.51 in  — 1.5 column
    "full": 170.0 * _MM,    # 6.69 in  — full page width
}
MAX_HEIGHT_IN = 228.0 * _MM  # 8.98 in — hard ceiling

# --------------------------------------------------------------------------- #
# Color — one visual law: warm = Abrupt, cool = Non-abrupt, everywhere.
# --------------------------------------------------------------------------- #
LOG_EVIDENCE_CMAP = _cmc.vik            # diverging; pale center = neutral (0)
DI_CMAP = _cmc.batlow                   # sequential (AOA dissimilarity index)

# Class anchors are sampled from vik's poles so the class dots visually rhyme
# with the diverging map's positive/negative ends (0.15/0.85, not the too-dark
# extremes 0.0/1.0).
NON_ABRUPT = to_hex(_cmc.vik(0.15))     # cool blue   (#044d87)
ABRUPT = to_hex(_cmc.vik(0.85))         # warm rust   (#a13c0b)

# Encoding is fixed: 0 = Abrupt (majority), 1 = Non-abrupt (minority).
CLASS_COLORS = {0: ABRUPT, 1: NON_ABRUPT}
CLASS_NAMES = {0: "Abrupt", 1: "Non-abrupt"}

# Support neutrals.
DOMAIN_GRAY = "#d9d9d9"                  # permafrost-domain backdrop
MASK_COLOR = "#9e9e9e"                   # out-of-AOA / masked base tint
MASK_HATCH = "////"                      # out-of-AOA is a HATCH, not a value color
INK = "#1a1a1a"                          # primary text / marks
MUTED = "#666666"                        # secondary text

# CVD-safe qualitative palette (Okabe-Ito) for incidental categoricals. The
# SHAP family palette (top-N + "Other") is designed at Fig 7/9 time (decision 4c);
# use OTHER_GRAY for the collapsed bucket.
QUALITATIVE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
OTHER_GRAY = "#bdbdbd"


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
_FONTS_REGISTERED = False


def register_fonts() -> None:
    """Register the in-repo Source Sans 3 faces with matplotlib (idempotent)."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    for ttf in sorted(_FONTS.glob("*.ttf")):
        _fm.fontManager.addfont(str(ttf))
    _FONTS_REGISTERED = True


def use() -> None:
    """Register fonts and apply the house stylesheet. Call once per script."""
    register_fonts()
    plt.style.use(str(_STYLE))


def figure(width: str = "single", *, aspect: float = 0.75, height: float | None = None,
           **kwargs):
    """Create a figure+axes at an AGU column width.

    width  : "single" | "onehalf" | "full" (mapped to AGU column inches)
    aspect : height / width ratio (used when `height` is not given)
    height : explicit height in inches (clamped to the 228 mm AGU ceiling)

    Returns (fig, ax). Pass ``subplots=False`` to get just the figure.
    """
    if width not in WIDTHS_IN:
        raise ValueError(f"width must be one of {sorted(WIDTHS_IN)}; got {width!r}")
    w = WIDTHS_IN[width]
    h = height if height is not None else w * aspect
    if h > MAX_HEIGHT_IN + 1e-6:
        raise ValueError(
            f"height {h:.2f} in exceeds AGU ceiling {MAX_HEIGHT_IN:.2f} in (228 mm)"
        )
    make_axes = kwargs.pop("subplots", True)
    if make_axes:
        return plt.subplots(figsize=(w, h), **kwargs)
    return plt.figure(figsize=(w, h), **kwargs)


# --------------------------------------------------------------------------- #
# Log-evidence: symmetric-about-zero normalization (enforced)
# --------------------------------------------------------------------------- #
def symmetric_norm(vmax: float) -> Normalize:
    """Normalize locked symmetric about 0 so vik's pale center == log-evidence 0.

    Never build a log-evidence color scale any other way: a drifting center
    would visually misstate which locations favor abrupt thaw.
    """
    vmax = abs(float(vmax))
    if not np.isfinite(vmax) or vmax == 0:
        raise ValueError(f"vmax must be a finite non-zero number; got {vmax!r}")
    return Normalize(vmin=-vmax, vmax=vmax)


def log_evidence_colorbar(mappable, ax=None, *, label="Log-evidence (abrupt vs. non-abrupt)",
                          **kwargs):
    """Colorbar for the log-evidence field with an explicit 0 tick and labeled poles.

    The pole labels are the one load-bearing text exception to the minimize-text
    rule: the reader must be told which end favors which mode, and that it is
    log-evidence, not probability.
    """
    fig = (ax.figure if ax is not None else plt.gcf())
    cbar = fig.colorbar(mappable, ax=ax, **kwargs)
    vmin, vmax = mappable.norm.vmin, mappable.norm.vmax
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.set_label(label, fontsize=8)
    cbar.ax.text(0.5, 1.02, "favors abrupt", transform=cbar.ax.transAxes,
                 ha="center", va="bottom", fontsize=6.5, color=MUTED)
    cbar.ax.text(0.5, -0.02, "favors non-abrupt", transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=6.5, color=MUTED)
    return cbar


# --------------------------------------------------------------------------- #
# Panel labels: parenthesized lowercase bold — (a) (b) (c)
# --------------------------------------------------------------------------- #
def panel_label(ax, letter: str, *, loc: str = "upper left", pad: float = 0.02):
    """Place a bold ``(a)``-style panel label. One helper so offset never drifts."""
    ha, va = ("left", "top")
    x, y = pad, 1.0 - pad
    if "right" in loc:
        x, ha = 1.0 - pad, "right"
    if "lower" in loc:
        y, va = pad, "bottom"
    return ax.text(x, y, f"({letter})", transform=ax.transAxes, ha=ha, va=va,
                   fontsize=9, fontweight="bold", color=INK)


# --------------------------------------------------------------------------- #
# North arrow (map furniture; scale bar is CRS-dependent, built at figure time)
# --------------------------------------------------------------------------- #
def north_arrow(ax, x: float = 0.95, y: float = 0.14, size: float = 0.09):
    """Draw a minimal N arrow in axes fraction coordinates."""
    ax.annotate("N", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                ha="center", va="center", fontsize=8, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))


# --------------------------------------------------------------------------- #
# Save: PDF canonical + PNG companion, from one call
# --------------------------------------------------------------------------- #
def save(fig, name: str, *, outdir: Path | None = None, rasterized_dpi: int = 300,
         png: bool = True, tight: bool = True):
    """Write ``<name>.pdf`` (canonical, vector) and ``<name>.png`` (viewing).

    Rasterized data layers (set ``rasterized=True`` on the heavy artist in the
    script) are flattened at ``rasterized_dpi`` (300 default; 600 for fine
    combination detail); vector text/axes stay crisp. Returns the PDF path.

    Set ``tight=False`` for multi-Axes figures with rasterized images (imshow /
    hexbin) in more than one Axes: the house style's ``savefig.bbox: tight``
    forces a two-pass PDF render, and matplotlib's mixed-mode PDF renderer
    mis-places rasterized images across that second pass when multiple Axes
    each hold one (they all collapse into one Axes' corner, scaled down, with
    other rasterized content going blank) — a real matplotlib limitation, not
    a bug in the figure script. Hand-tune margins with ``subplots_adjust``
    instead of relying on tight-bbox cropping when you pass ``tight=False``.
    """
    outdir = Path(outdir) if outdir is not None else _HERE
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    with mpl.rc_context({} if tight else {"savefig.bbox": None}):
        fig.savefig(pdf, dpi=rasterized_dpi)
        if png:
            fig.savefig(outdir / f"{name}.png", dpi=300)
    return pdf


# --------------------------------------------------------------------------- #
# CVD validator — enforces the accessibility constitution
# --------------------------------------------------------------------------- #
# Machado, Oliveira & Fernandes (2009) severity-1.0 simulation matrices,
# applied to linear-light RGB.
_CVD_MATRICES = {
    "deuteranopia": np.array([[0.367, 0.861, -0.228],
                              [0.280, 0.673, 0.047],
                              [-0.012, 0.043, 0.969]]),
    "protanopia": np.array([[0.152, 1.053, -0.205],
                            [0.115, 0.786, 0.099],
                            [-0.004, -0.048, 1.052]]),
}


def _srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _rgb_to_lab(rgb):
    lin = _srgb_to_linear(rgb)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def simulate_cvd(rgb, kind: str):
    """Simulate how `rgb` (0-1) appears under `kind` dichromacy."""
    lin = _srgb_to_linear(rgb)
    return _linear_to_srgb(lin @ _CVD_MATRICES[kind].T)


def min_delta_e(colors) -> dict:
    """Min pairwise CIELAB ΔE (1976) among `colors` under normal + CVD vision."""
    rgb = np.array([to_rgb(c) for c in colors])
    out = {}
    for kind in ("normal", "deuteranopia", "protanopia"):
        sim = rgb if kind == "normal" else simulate_cvd(rgb, kind)
        lab = _rgb_to_lab(sim)
        best = np.inf
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                best = min(best, float(np.linalg.norm(lab[i] - lab[j])))
        out[kind] = best
    return out


def assert_cvd_safe(colors, *, min_de: float = 15.0, name: str = "palette") -> dict:
    """Raise if any two `colors` collide under normal or simulated CVD.

    ΔE ≈ 15 is a conservative "clearly distinct" floor for categorical marks.
    """
    de = min_delta_e(colors)
    worst = min(de.values())
    if worst < min_de:
        raise AssertionError(
            f"{name} not CVD-safe: min ΔE {worst:.1f} < {min_de} "
            f"(by-vision: {', '.join(f'{k} {v:.1f}' for k, v in de.items())})"
        )
    return de


def validate() -> None:
    """Self-check the palettes that ship with the module."""
    pair = assert_cvd_safe([ABRUPT, NON_ABRUPT], min_de=25, name="class pair")
    qual = assert_cvd_safe(QUALITATIVE, min_de=15, name="qualitative (Okabe-Ito)")
    print("class pair CVD ΔE:", {k: round(v, 1) for k, v in pair.items()})
    print("qualitative CVD ΔE:", {k: round(v, 1) for k, v in qual.items()})
    print("OK — palettes are CVD-safe.")


if __name__ == "__main__":
    validate()
