# Figure 8 redesign spec — SHAP mechanism (own-SHAP dependence)

**Status:** ready for sub-agent handoff. Decided in the 2026-07-23 grilling session.
**Target script:** rewrite `output/fig08_shap_mechanism.py` in place (same output names).
**Outputs:** `output/08_shap_mechanism.{pdf,png}` (unchanged filenames).
**Do NOT touch:** `models/shap_groups.py`, `models/shap_mechanism_cache.py`, the family
definitions, or the importance metric. The grouping and the `mean|Σ member SHAP|` metric are
**sound and stay** — this is a *presentation* fix only (see "Why" below). Do not edit any `.tex`.

---

## Why we're changing it (context the sub-agent must preserve)

The old Fig 8 plotted, per family, the **family-summed SHAP (y) against the single leading
member's value (x)**. Two problems:

1. **Coordinate contamination.** y was the whole family's sum but x was only one member, so a
   panel's shape was partly driven by the *other* members. It also inflated single-feature
   claims: e.g. at high slope the family-sum reads ~+1.1, but ~+0.6 of that is
   Elevation/HAND/Mean-Annual-SWE piling on — Slope's *own* effect is only ~+0.45. You could
   not make an honest "slope > X → abrupt" statement from it.
2. **Manufactured climate-heavy look.** The old "family importance ≥ 5%" panel-selection rule
   was cleared more often by climate purely because climate is split into more families, so
   4 of 5 continuous panels were climate. That created a false "climate dominates" gestalt.

**Fix:** plot each feature's **own SHAP vs its own value** (the standard dependence plot), for
a curated set of the most influential *individually interpretable* features. This (a) supports
honest per-feature threshold statements, (b) removes the contamination, and (c) auto-fixes the
climate gestalt — selected by raw influence, the roster is multi-domain (4 relief / 2 snow /
3 climate), with relief supplying the plurality, matching the family ranking in Fig 7.

The magnitude/importance story stays in Fig 7 (the grouped families). Fig 8's job is **shape
only** ("how"). Land Cover is **no longer in Fig 8** — it becomes its own separate figure.

## Framing discipline (locked)

Report shapes as **fact about the model's response**, never as asserted mechanism. Captions:
"the model's evidence for abrupt thaw rises above ~12° slope," NOT "steep slopes cause abrupt
thaw." Mechanistic interpretation and the proxy-vs-mechanism defense are reserved for §5.2 /
Fig 9. This matches `manuscript/OUTLINE.md` ("shapes reported as fact").

---

## Data

- `output/shap_mechanism_cache.npz` (already written by `models/shap_mechanism_cache.py`):
  - `values` (n, F) — per-feature OOF SHAP, **Abrupt-oriented** (positive ⇒ favors Abrupt).
  - `data` (n, F) — feature values.
  - `feature_names` (F,) object array.
  - `y` (n,) — labels (0 = Abrupt, 1 = Non-abrupt).
  - n = 19,288 points, F = 70 features. No regeneration needed unless the model/features change.
- `output/shap_families.json` — used only for optional family/domain labeling; **selection is
  now per-feature**, not per-family.
- **Sign-convention check (repo rule):** before finalizing, confirm the Abrupt orientation is
  correct — cold Annual Mean Temperature should read **positive** (favors Abrupt). If it reads
  negative, the cache orientation is flipped; stop and flag rather than recoloring.

## Roster — 9 panels, 3×3 grid

Top-9 continuous features by own `mean|SHAP|` (values below for validation). Fixed order
(reading order, left→right, top→bottom):

| # | feature | own mean\|SHAP\| | domain | x display units |
|---|---|---|---|---|
| 1 | Slope | 0.93 | relief | ° (natural) |
| 2 | Annual Mean Temperature | 0.85 | climate | °C — **rescale ×0.1** (WorldClim tenths) |
| 3 | Trend in SWE | 0.37 | snow | mm yr⁻¹ |
| 4 | Isothermality | 0.34 | climate | % |
| 5 | Mean Annual SWE | 0.33 | snow | mm |
| 6 | Height Above Nearest Drainage | 0.24 | relief | m |
| 7 | Trend in precipitation | 0.23 | climate | mm yr⁻¹ (confirm units in feature table) |
| 8 | Mean curvature (500 m) | 0.18 | relief | (curvature units; confirm) |
| 9 | Upstream Area | 0.15 | relief | km² (confirm; may need log x if tail crushes mass) |

The roster is deliberate — do not re-select by a family cutoff. If regeneration ever changes
the ranking, recompute own `mean|SHAP|` and keep the top-9 interpretable continuous features,
but flag any change to the human first.

## Per-panel design

Each panel = one feature's **own SHAP (y) vs its own value (x)**:

- **Density:** hexbin, neutral gray, `bins="log"`, `mincnt=1`, rasterized — same as old fig
  (so it never competes with the sign colors).
- **Trend:** running median over fixed-width x-bins (reuse `N_XBINS=22`, `BIN_N_MIN=40`) with a
  25–75% band. Sign-color the trend segments: **warm (`figstyle.ABRUPT`) where median ≥ 0
  (favors Abrupt), cool (`figstyle.NON_ABRUPT`) where < 0** (favors Non-abrupt).
- **Zero line:** horizontal `axhline(0)`, thin ink. Crossing not annotated.
- **x-range:** robust 0.5–99.5 percentile clip (as old fig). Slope floored at 0.
- **Per-panel annotation:** feature's own `mean|SHAP|` and its rank (small, muted) — so the
  reader knows the panels differ in magnitude and where to look in Fig 7 for importance.
- **Titles:** short feature name. Two-line x-labels with units where needed.

### Y-axis (default chosen — confirm with human if weak panels are illegible)

**Shared symmetric y across all 9 panels** (`±ymax` from robust percentiles across the 9
own-SHAP arrays), consistent with the figure-style-guide symmetric lock and honest about
magnitude (weaker features *should* look flatter — that's true). If on render the bottom-row
panels (curvature, upstream) are too flat to read their shape, fall back to per-panel y with a
prominent own-`mean|SHAP|` annotation, and note the change back to the human.

### Edge-artifact handling (locked)

1. **Slope-0 point mass** (~40% of points at *exactly* 0°, own-SHAP median ≈ +1.77, the
   strongest *and* most sampling-biased signal — "flat" ≈ lake/lowland where abrupt was
   sampled). Peel it out: draw `slope == 0` as a **distinct marker** (median + IQR) at x = 0,
   and **start the running-median trend at the smallest positive slope bin** with a visible
   break, so the continuous gradient for slope > 0 is legible and the flat-ground signal is
   shown but visibly separate. Exclude the 0-mass from the trend bins.
2. **Temperature warm cliff** (Annual Mean Temperature > −4 °C plunges to ≈ −2.4; ~5% of
   points, permafrost-domain boundary). Keep it, but annotate as a small warm-boundary
   population (light shading of that x-region or an "≈5%, domain edge" note).
3. **Caption line:** state that the flat-slope and warm-edge extremes are the most sampling-/
   boundary-sensitive regions — pre-empting the proxy concern where a reader would raise it.

## Expected shapes (validation targets — the render should reproduce these)

- **Slope:** non-monotonic, two-regime. Flat (0°) ≈ +1.77 (abrupt); gentle 0–6° dips to ≈ −0.25
  (non-abrupt); steep >12° rises to ≈ +0.45 (abrupt). NOT a simple monotonic threshold.
- **Annual Mean Temperature:** cold (−15 to −5 °C) steadily ≈ +0.6 to +1.1 (abrupt); warm edge
  (> −4 °C) cliff to ≈ −2.4 (non-abrupt).
- Others: report the shape as found; no prior expectation locked.

## Figure mechanics

- Pure plotting; reads the cache. Use `figstyle` (`figstyle.use()`, `figstyle.save`, AGU full
  width `figstyle.WIDTHS_IN["full"]`, `figstyle.ABRUPT`/`NON_ABRUPT`, panel labels a–i).
- 3×3 `GridSpec`. Panel labels "abcdefghi".
- Directional cue stated once at the figure foot: "warm → favors Abrupt" / "favors Non-abrupt
  ← cool".
- Print a summary line on run (roster + shared-y value + any fallback taken).

## Out of scope for this handoff (separate figures)

- **Land Cover per-class decomposition** → its own new main figure (own-SHAP not applicable; it
  is a categorical box-per-class of the Land Cover *family* SHAP). Spec separately.
- **Fig 7** (families magnitude + direction) → separate redesign to stop the cross-family
  climate subtotal. Not this handoff.
- **Fig 9** (dominance map) → unbuilt, design-forward, separate session.
