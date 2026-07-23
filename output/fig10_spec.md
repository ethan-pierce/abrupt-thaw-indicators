# Figure 10 spec — SHAP dominant-domain map (L6c / L7)

**Status:** ready for sub-agent handoff. Decisions closed in the 2026-07-23 grilling session.
**NEW figure** — script `output/fig10_shap_dominance.py`; outputs `output/10_shap_dominance.{pdf,png}`.
**Heavy compute** — per-cell TreeSHAP over the in-AOA grid; smoke on a subsample first.
**Do NOT touch:** `models/*` training code, family/metric definitions. Do not edit any `.tex`.

---

## Purpose

Map, for each in-AOA grid cell, **which thematic domain dominates the model's evidence there.**
Its job is the **proxy rebuttal** (L6c → §5.2): if the dominant driver varies regionally, the
signal is not one smooth spatial trend a single location-proxy would produce. **Descriptive only**
— all-data model, display restricted to AOA, no calibrated-probability or causal claims.

## Decisions (all locked — do not re-open)

- **Unit:** dominant **domain** per cell (not family). Same 8 thematic domains as Fig 7.
- **"Dominant" metric:** per cell, for each domain sum its features' per-cell SHAP → the domain's
  net contribution; the cell's class = the domain with the largest **|net contribution|**
  (unsigned — *which kind of driver moves this cell's prediction most*, regardless of direction).
  Direction/susceptibility is Fig 4's job, not this map's.
- **Encoding:** **hue only** (flat categorical fill = dominant domain). No saturation/alpha
  modulation (rejected as visually hard to read). Winner-take-all; honesty about near-ties is
  carried by the area-fraction inset + caption, not by shading.
- **Area-fraction inset:** include — a small companion bar of *% of in-AOA area each domain
  dominates.* Quantifies the map's point and doubles as the validation gate (below).
- **Palette:** the SAME domain colors as Fig 7 (from the shared source below). Legend scoped to
  domains that actually dominate a non-trivial share of cells; fold negligible ones into "other".

## VALIDATION GATE (run before finalizing the figure)

Compute the area fractions FIRST. **If a single domain dominates more than ~60% of in-AOA area,
the map is near-monochrome and the proxy-rebuttal framing is weakened — STOP and flag for the
human** (we agreed to rethink the figure's framing in that case rather than force it). Report the
full area-fraction table on the smoke run so this is caught before the full compute.

## Single source of truth for domains (create this; Fig 7 imports it too)

Create `output/shap_domains.py` holding (a) the **feature → domain** assignment and (b) the
**domain color palette**, so Fig 7 and Fig 10 are guaranteed identical. Feature→domain map (final,
from the Fig 7 spec — assign one-hot `Land Cover (*)` / `Vegetation Mode (*)` to Land cover FIRST,
before any keyword match):

- **Snow** (2): Mean Annual SWE, Trend in SWE
- **Seasonality** (2): Isothermality, Precipitation Seasonality
- **Temperature** (11): Annual Mean Temperature, Mean Temperature of Driest Quarter, Mean Diurnal
  Range, Temperature Seasonality, Min Temperature of Coldest Month, Temperature Annual Range, Mean
  Temperature of Coldest Quarter, Max Temperature of Warmest Month, Mean Temperature of Wettest
  Quarter, Mean Temperature of Warmest Quarter, Trend in temperature
- **Precipitation** (8): Annual Precipitation, Precipitation of Wettest/Driest Month,
  Precipitation of Wettest/Driest/Warmest/Coldest Quarter, Trend in precipitation
- **Relief** (8): Slope, Elevation, Height Above Nearest Drainage, Mean curvature (500 m), Mean
  curvature (2 km), Upstream Area, Eastness, Northness
- **Soil** (10): Sand ×2, Clay ×2, Bulk Density ×2, Soil Organic Carbon ×2, Nitrogen ×2
- **Land cover / veg / fire** (28): all `Land Cover (*)` + all `Vegetation Mode (*)` +
  Flammability Index + Time Since Last Fire + Burn Count
- **Other / ground ice** (1): Yedoma

The palette must be CVD-safe and distinct from `figstyle.ABRUPT`/`NON_ABRUPT`. (If Fig 7 was built
before this file existed, refactor its inline map to import from here.)

## Compute recipe

Inputs (verified paths):
- Datacube: `data/prediction_data.nc` — `feature_stack` (EPSG:4326, 1 km grid, per-cell lon/lat
  coords). 3.9 GB — process in chunks.
- Model: `models/model.json` (all-data XGBoost). Feature order = `learner.feature_names` in the
  JSON (predict.py reads it this way — match it exactly).
- AOA: `data/aoa.nc` (`DI` dissimilarity index) + threshold in `models/aoa_threshold.json`
  (DI ≤ threshold ⇒ in-AOA). ~2,773,804 in-AOA cells expected.

Steps:
1. Reshape `feature_stack` to `(n_cells, 70)` in the model's feature order; select in-AOA cells.
2. Per-cell SHAP via XGBoost `pred_contribs` (Booster `predict(..., pred_contribs=True)` or
   `shap.TreeExplainer`), in **chunks** (e.g. 200k cells) to bound memory (full contribs ≈ 2.8M×70
   floats ≈ 1.6 GB). Orientation is irrelevant here — we take **|domain sum|**, so no sign flip is
   needed (unlike Fig 7/8). Drop the bias column.
3. Aggregate the 70 per-feature contributions to the 8 **domain** net contributions (sum within
   each domain, using `shap_domains.py`).
4. `argmax` over domains of `|domain net contribution|` → dominant-domain code per cell.
5. **Cache** the dominant-domain raster + the area-fraction table to
   `output/shap_dominance_cache.npz` (mirrors the diagnostics→cache→figure pattern; the figure
   script is then pure plotting and re-runs fast). A `SHAP_DOM_SMOKE=1` env flag should subsample
   for a wiring check, writing to `output/_smoke/`.

## Map rendering (mirror Fig 4 / Fig 2)

- Warp the EPSG:4326 grid into **Alaska Albers (EPSG:3338)** using the affine derived from the
  lon/lat coords, exactly as `output/fig04_susceptibility_aoa.py` does (reuse that warp code).
- **Display in-AOA cells only**; render out-of-AOA (and non-permafrost) as neutral gray, consistent
  with Fig 4's reliability treatment. Draw the permafrost-domain (Obu) boundary + coastline to
  match Figs 2/4.
- Categorical fill = dominant-domain color (shared palette). Legend lists the dominant domains.
- **Area-fraction inset:** small horizontal bars, domain-colored, `% of in-AOA area dominated`,
  sorted descending. Same colors/labels as the map legend.
- Use `figstyle` (`use()`, `save`, AGU width). Print the area-fraction table + the validation-gate
  verdict on run.

## Framing discipline (locked)

Descriptive attribution, not mechanism or necessity. Caption states: all-data model, AOA-restricted
display, "dominant" = largest |domain contribution|, and that regional variation in the dominant
driver is evidence *against* a single smooth location-proxy (the §5.2 argument) — without claiming
causation.

## Coherence note (caption)

Fig 7 ranks families/domains from **OOF fold-refit** SHAP on the 19,288 training points; Fig 10 maps
the **all-data** model's SHAP over the grid. Acceptable per STRATEGY.md; the **domain definitions
are identical** (both import `shap_domains.py`). Family/domain grouping is from feature correlation,
not SHAP (anti-circularity), stated once in Methods.
