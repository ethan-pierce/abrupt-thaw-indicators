# Manuscript figure list

Working figure spec for the map-led Earth's Future draft. Derived from the L1→L7
argument chain in `STRATEGY.md` (figures fall out of the chain, not vice versa).
**11 main figures**, in manuscript order. Each is handed off to a sub-agent one at a
time; this file is the shared source of truth.

Class encoding (fixed): `0 = Abrupt` (majority ~93%), `1 = Non-abrupt` (minority ~7%);
metric "positive" = Non-abrupt. Index = prior-free **log-evidence** (`>0` favors abrupt),
never a calibrated probability. Language discipline: "features more consistent with
abrupt thaw" — never "% susceptible" / "% will thaw" / probability.

## Setup (L1 + methods)

1. **Concept figure** — *L1* · hand-drawn + photos
   - (a) schematic of abrupt vs non-abrupt thaw pathways + the mode vs stage vs occurrence distinction (the novelty hook: predicting *mode*, which no incumbent does).
   - (b) field-photo plate of abrupt-thaw forms (thermokarst lakes, retrogressive thaw slumps, etc.). Presumes licensed/publishable photos.
   - Not script-generated (Illustrator); a sub-agent specs/mocks it.

2. **Study-area / sampling map** — *L1* · new (script)
   - Alaska permafrost domain + the 19,288 deduplicated model-training locations, shown as matched 25-km, shared-log-scale density hexagons for abrupt and non-abrupt classes. The clustering is explicit; add sourced roads/hydrography only if a citable statewide layer is acquired. The clean lead makes the bias story visible for L3/L4c.

3. **Methods / pipeline schematic** — hand-drawn
   - features → XGBoost → SHAP → log-evidence → map + AOA. Helps readers follow the unusual log-evidence framing. Not script-generated; sub-agent specs/mocks it.

## Headline

4. **Susceptibility map + AOA mask panel** — *L5 + L4b*
   - Assets: `output/susceptibility_log_evidence_map.png` + `output/aoa_map.png`.
   - The product. Headline: ~25.6% of the in-AOA permafrost domain has features more consistent with abrupt than non-abrupt thaw (710,882 of 2,773,804 in-AOA cells, log-evidence > 0). Anchored only to log-evidence = 0. AOA merged as a panel so the product is never shown without its reliability mask (AOA flags 2.7% of the valid domain).

## Credibility

5. **Spatial out-of-sample performance** — *L2a, L2c, L3* (MERGED) · script: `output/fig05_spatial_performance.py`
   - Spine: the signal is **not a location proxy** — it generalizes across space. Two panels, each refuting a distinct proxy attack, both scoring the **same operative model** (spw=1, selected hparams) under progressively harder spatial regimes. Colored value ladder (XGBoost blue / logistic orange-dashed / gray-dotted floor); the floor is the only reference anchor (no leaky ceiling). Numbers annotated on-figure are read live from the cached JSON.
   - (a) **block-size ladder** (refutes short-range leakage): repeated spatial block-CV, 20 reshuffles/scale (`diagnostics/repeated_cv.py` → `output/repeated_cv_results.json`), laddered floor → logistic → XGBoost. Headline: **AUC-PR 0.852 ± 0.011 @ 10 km** (repeated-CV mean ± across-partition σ), ≈15× the 0.0574 floor; logistic 0.776, margin +0.076 ± 0.019. Caveat (caption): hyperparameters fixed, so per-fold selection cost is not re-paid.
   - (b) **leave-region-out extrapolation** (refutes region memorization): AUC-PR vs median distance-to-nearest-training-point (`diagnostics/extrapolation_range.py` → `output/extrapolation_range_results.json`), graceful decay to **0.54 @ 251 km** (3 held-out regions, ~9× floor).
   - `diagnostics/leakage_decay.png` (buffer sweep) is demoted to the Supplement (see below); `output/aucpr_vs_blocksize.png` is superseded.

6. **Representativeness / sampling-bias honesty gate** — *L4c* · script: `output/fig06_representativeness.py`
   - The honesty gate, framed as *scope not defect*: the training sample is lake-/road-biased (flatter/lower/wetter/valley-bottom than the statewide grid), which forbids prevalence / calibrated-probability / single-threshold claims (hence the prior-free log-evidence index, L4a) while the discriminative signal itself generalizes across space (Fig 5 / L3). **Coverage (the AOA, Fig 4b) vs density (this figure) are different things** — a sample can span every covariate's full range (so every cell is in-AOA) while wildly over-representing part of it; the AOA can't see that, this figure shows it.
   - Marginal distributions, **training sample vs in-AOA statewide grid** (2,773,804 cells — the full in-AOA distribution, *not* the matched-cell parity sample), for 7 cherry-picked features. (a–d) train-over-grid ridgelines on an arcsinh (symlog-consistent) x, median-annotated: Slope (0.74° vs 3.7°), Height Above Nearest Drainage (1.0 m vs 16 m), Elevation (61 m vs 280 m), Upstream Area (0.023 vs 0.0099 km²). (e–g) paired bars: Open Water (0.43 vs 0.039), Emergent Herbaceous Wetland (0.076 vs 0.037), Deciduous Forest (0.010 vs 0.036).
   - Green = training, purple = in-AOA grid (deliberately off the warm/cool class axis, CVD-checked); quantitative annotations only (distributions + medians, no fold-change or semi-qualitative text); invariance argument kept lean (conceptual in text; `s(x)`-cancellation parked in Methods/Supplement).

## Interpretation (SHAP)

7. **SHAP global + emergent families** — *L6a + L6b* · script: `output/fig07_shap_families.py` (reads `output/shap_grouped_matrix.npz`, written by `models/shap_groups.py`)
   - Opens §4.3 by *establishing the analysis unit before attributing* — the family construction is a result (grouping is from feature-space correlation, not SHAP → preempts circularity). That anti-circularity point is made in prose/caption; the **dendrogram itself is a Supplement figure** (`output/shap_family_dendrogram.png`), not shown here — at main-text size its 44 leaf labels are unreadable, and it can't represent the collapsed categorical families (Land Cover etc.) at all.
   - 2-panel, sharing one **importance-sorted family y-axis** (all 22 families, most important on top): **(a) magnitude** — horizontal teal bars, mean over points of `|Σ member SHAP|` (margin), annotated with each family's % of summed family importance; **(b) signed direction** — per-family **zero-split violin** (mass right of 0 warm = favors Abrupt, left cool = favors Non-abrupt), KDE clipped to the observed data range, with median + 5/95 marks overlaid; vertical zero line; directional cue flanking the x-axis.
   - Both panels earn their place: **Land Cover** is #3 by magnitude yet its signed distribution *straddles zero* (median ≈ 0.02, p5/p95 = −0.20/+1.35) — it discriminates in both directions, which a mean-signed bar would erase.
   - Grouped families (Abrupt-oriented, OOF fold-refit SHAP): Alpine relief 23%, Annual/dry-season temperature 16%, Land Cover 12%, Thermal continentality 9%, … Fire history ranks **last** (<1%) — an informative null kept visible in the tail.
   - Family palette for the Fig 9 dominance map is deferred to Fig 9 (scoped to the families that actually dominate a cell), not born here.

8. **SHAP mechanism** — *L6b (deepened)* · new
   - Dependence plots (SHAP value vs underlying feature value) for the top ~4 families — shows *how* each family pushes, not just which way. This is what earns the reserved mechanistic language (alpine relief → ground-ice/drainage; temperature → thermal state).

9. **SHAP-dominance map** — *L6c / L7* · new
   - Per-cell TreeSHAP over ~2.85M in-AOA cells (`models/model.json`), mapped as the dominant grouped family per cell. Descriptive only (all-data model, restrict display to AOA). Doubles as the proxy rebuttal — regionally varying drivers ≠ one smooth spatial trend.

## Downstream (L7)

10. **Ecoregion breakdown** — *L7* · new
    - Abrupt-favoring fraction by physiographic ecoregion (Alaska Unified Ecoregions); drop ecoregions below a permafrost-coverage threshold; report per-region AOA coverage; in-AOA only.

11. **Olefeldt incumbent contrast** — *L7* · new, **GATED**
    - Against Olefeldt et al. 2016 thermokarst-landscape classes (the only Alaska-statewide comparable incumbent). Reproject to grid; show log-evidence spans a wide range within a single Olefeldt class → mode is an orthogonal axis the categorical map doesn't resolve. **Positioning, NOT validation** — complementary/refining, never corrective; state plainly what Olefeldt maps (landscape type + occurrence potential, a valid different axis) to preempt the strawman concern. Data acquired (`data/Circumpolar_Thermokarst_Landscapes/`) → **Form B is the plan**; Form A (qualitative paragraph) remains the fallback if alignment proves hard. Form C (divergence-hotspot map) ruled out.

## Supplement

- L2b artifact controls: shuffle → chance, dummies at floor, contradictory-label count (only 4 pairs).
- Train/serve **construction**-parity gate (`diagnostics/train_serve_parity.{png,md}`): per-feature agreement of the training column vs the datacube pixel at matched cells (Spearman ρ / 0–1 agreement) — a QA artifact proving no unit/transform slip. Demoted here from the old Fig 6 slot: it answers "is the pipeline wired right?", a different question from Fig 6's *density* / representativeness story (the offset-sensitive features surface the sampling bias only as a byproduct).
- Leakage-decay buffer sweep (`diagnostics/leakage_decay.png`): finer-grained short-range control for Fig 5a; its right half degenerates once the training pool is depleted, so it is a Supplement control rather than a main panel.
- **SHAP family dendrogram** (`output/shap_family_dendrogram.png`) + full family membership table + clustering parameters. The dendrogram lives here, not in main-text Fig 7: at column size its 44 leaf labels are unreadable and it can't show the collapsed categorical families. The Supplement carries the tree, the full per-family member list, and the cut-threshold detail; apply family definitions identically to L6a and L9.
- AOA dissimilarity-index detail: `output/aoa_di_map.png`, `output/aoa_threshold_decision.png`.
- Calibration: `diagnostics/aoa_calibration.png`.
- Extra / unused field photos.

## Key stamps (from FINDINGS.md)

70 features × 19,288 rows; prevalence 93.21/6.79 (positive = Non-abrupt, floor 0.0574);
model `models/model.json`; CV = albers_grid 10 km, buffer 0.0, 5×5 nested, seed 42;
grid 3229×2087, 2,849,807 valid cells (42.3% of grid), 2,773,804 in-AOA.

## TODO

- Confirm datacube cell size to convert 25.6% → absolute km² for the abstract.
