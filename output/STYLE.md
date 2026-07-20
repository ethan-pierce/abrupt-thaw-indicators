# Figure style guide

House style for every figure in the **abrupt-thaw-indicators** manuscript
(target: *Earth's Future*, AGU / Wiley). The goal is that all 11 main figures
plus the supplement read as **one system** — same palette, type, sizing, and
language — so a reviewer never has to re-learn how to read a figure.

Two layers:

- **`figstyle.py` + `abrupt_thaw.mplstyle`** — the machine-enforced layer. Import
  it and the palette, colormaps, fonts, sizes, and output format come for free.
- **This document** — the judgment layer: the rules code can't enforce
  (composition, text discipline, map furniture, captions).

> **The existing figures carry no authority.** Everything under `output/*.png`
> and `diagnostics/*.py` predates this guide and is early-draft inventory. Do not
> copy their palette, sizing, or layout. Restyle each one against this guide when
> it is promoted toward the manuscript.

---

## How to use it

Every figure script:

```python
import figstyle

figstyle.use()                                   # apply style + register fonts
fig, ax = figstyle.figure("single", aspect=0.8)  # canvas at a real AGU width
ax.scatter(x, y, color=figstyle.CLASS_COLORS[0]) # 0 = Abrupt = warm
figstyle.panel_label(ax, "a")                    # -> (a), bold, top-left
figstyle.save(fig, "04_susceptibility_map")      # 04_...pdf (canonical) + .png
```

Scripts live in `output/`. When a render is ready, **copy** (never move) the
final asset into `manuscript/figures/` with a two-digit order prefix
(`04_susceptibility_map.pdf`); generation scripts stay in `output/`.

Run `poetry run python figstyle.py` to self-check the palettes are CVD-safe.

---

## Decision record

| # | Decision | Resolution |
|---|----------|------------|
| 1 | Form | Code-first: shared module + `.mplstyle` + this prose guide. |
| 2 | Venue | Bound to AGU/Wiley column widths (single 85 mm, full 170 mm, height ≤ 228 mm). |
| 3 | Color constitution | CVD-safe + perceptually uniform, **mandatory & validated**. Continuous fields = Crameri Scientific Colour Maps. |
| 4 | Categorical | Warm = Abrupt, cool = Non-abrupt (anchored to `vik`'s poles). Domain = neutral gray; out-of-AOA = **hatch**. Family cap (top-N + "Other") set at Fig 7/9. |
| 5 | Log-evidence map | `vik`; normalized **symmetric about 0**; single shared fixed `vmax`. |
| 6 | Typography | Source Sans 3 (vendored). Scale: tick 7 / axis 8 / subtitle 9 semibold / panel-letter 9 bold / annotation 6.5 pt. Panel letters `(a) (b) (c)`. |
| 7 | Output | **PDF canonical + 300-dpi PNG companion**; rasterize heavy data layers at 300 dpi; embed TrueType fonts (`fonttype 42`). |
| 8 | Conventions | Provenance/description in **caption only**; no in-figure titles; **minimize in-figure text**; mandatory map furniture. |

---

## Rules the code can't enforce

### Text discipline
- **No descriptive title baked into a figure.** The caption carries the title
  and all description. Single-panel figures get no title text at all.
- **Minimize in-figure text.** This is the governing aesthetic: if a label,
  annotation, or subtitle isn't load-bearing, cut it and let the caption do the
  work. Multi-panel *subtitles* are allowed but preferred omitted.
- **Panel letters are always present** — `(a) (b) (c)`, via
  `figstyle.panel_label`. Never hand-roll them.
- **Provenance lives in the caption, never in the figure** — no source/credit
  line drawn on the canvas.

### Language discipline (hard rule)
All in-figure text — axis labels, legends, annotations, colorbars — uses
**log-evidence** framing:

- ✅ "log-evidence (abrupt vs. non-abrupt)", "features more consistent with
  abrupt thaw", "favors abrupt / favors non-abrupt".
- ❌ "probability", "% susceptible", "% will thaw", "P(abrupt)".

The index is a prior-free log-evidence ratio (`0` = neutral, `>0` favors
abrupt) — **not** a calibrated probability. The colorbar reads *log-evidence*.

### Color law
- **Warm = Abrupt, cool = Non-abrupt — everywhere.** Class dots
  (`figstyle.CLASS_COLORS`), the diverging map's poles, and SHAP direction all
  rhyme because the class colors are sampled from `vik`'s ends.
- Class encoding is fixed: **`0 = Abrupt` (warm), `1 = Non-abrupt` (cool)**.
  Verify indexing whenever you touch labels or `predict_proba`.
- **Continuous fields use Crameri maps only** — `figstyle.LOG_EVIDENCE_CMAP`
  (`vik`) for log-evidence, `figstyle.DI_CMAP` (`batlow`) for the AOA
  dissimilarity index. No `jet`, no `viridis`-by-habit, no bespoke gradients.
- **Log-evidence is always symmetric about 0** — build the norm with
  `figstyle.symmetric_norm(vmax)` so the pale center sits exactly on 0. Use one
  shared `vmax` across every figure that shows the field (Figs 4, 9, 10, 11) so
  color is comparable. The exact `vmax` (~99th pct of |log-evidence|) is fixed
  when Fig 4 is built.
- **Out-of-AOA / masked cells are a hatch overlay** (`figstyle.MASK_HATCH`),
  never a color on the data scale — so "unreliable" survives grayscale and is
  never mistaken for a value. The product is never shown without its AOA mask.
- **Family palette:** color only the top-N families (decision at Fig 7/9), rest
  collapse to `figstyle.OTHER_GRAY`. Same legend for the importance bars (Fig 7)
  and the dominance map (Fig 9).

### Map furniture (every statewide map)
- Graticule with degree labels.
- Scale bar (built at figure time — it depends on the map projection/CRS).
- North indicator (`figstyle.north_arrow`).
- Explicit projection note in the caption.
- Log-evidence colorbar via `figstyle.log_evidence_colorbar`: explicit **`0`
  tick** and labeled poles ("favors abrupt" / "favors non-abrupt"). These pole
  labels are the one text exception to *minimize-text* — they are load-bearing.

### Sizing
- Pick a canvas with `figstyle.figure("single" | "onehalf" | "full")` — these
  map to real AGU column inches, so **point sizes are literal printed sizes**.
- Never exceed 228 mm height (the helper enforces it).
- Set `rasterized=True` on the heavy data artist (map fields, dense scatter) so
  the PDF keeps vector text over a compact 300-dpi raster layer.

---

## Quick reference

**Colors** — `figstyle.CLASS_COLORS` `{0: ABRUPT #a13c0b, 1: NON_ABRUPT #044d87}`,
`DOMAIN_GRAY`, `MASK_COLOR` / `MASK_HATCH`, `INK`, `MUTED`, `QUALITATIVE`
(Okabe-Ito), `OTHER_GRAY`.

**Colormaps** — `LOG_EVIDENCE_CMAP` (vik), `DI_CMAP` (batlow).

**Helpers** — `use()`, `figure(width, aspect=/height=)`, `symmetric_norm(vmax)`,
`log_evidence_colorbar(mappable, ax)`, `panel_label(ax, "a")`,
`north_arrow(ax)`, `save(fig, name)`, `assert_cvd_safe(colors)` / `validate()`.

**Widths** — `WIDTHS_IN["single"|"onehalf"|"full"]` = 3.35 / 5.51 / 6.69 in;
`MAX_HEIGHT_IN` = 8.98 in.

**Type scale (pt, literal)** — tick 7 · axis 8 · subtitle 9 semibold ·
panel-letter 9 bold · annotation 6.5.
