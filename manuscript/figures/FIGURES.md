# Manuscript figure list

Working figure spec for the map-led Earth's Future draft. Derived from the L1→L7
argument chain in `STRATEGY.md` (figures fall out of the chain, not vice versa).
**12 main figures**, in manuscript order (the SHAP interpretation block is 7, 8, 9, 10).
Each is handed off to a sub-agent
one at a time; this file is the shared source of truth.

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

7. **SHAP family importance** — *L6a* · script: `output/fig07_shap_families.py` (reads `output/shap_grouped_matrix.npz` for family bars + `output/shap_mechanism_cache.npz` for the domain inset; see `output/fig07_redesign_spec.md`)
   - Opens §4.3 by *establishing the analysis unit before attributing* — the 22 emergent families are built from feature-space correlation, not SHAP (preempts circularity), stated in prose/caption. The **dendrogram is a Supplement figure** (`output/shap_family_dendrogram.png`).
   - **Single panel** (the direction violins are dropped — see below): horizontal bars, all 22 families, importance-sorted, `mean|Σ member SHAP|` (margin), each **colored by thematic domain** and annotated with its **member count** + % of summed family importance.
   - **Anti-subtotal device — a domain-aggregate inset:** the seven climate-ish family rows invite an illegitimate mental sum, so a small inset strip shows the *correct* feature-level thematic totals (`mean|Σ|` per domain): **Relief 24.7% ≈ Temperature 24.6%** (co-lead), Snow 14.3%, Land cover 14.3%, Precipitation 9.1%, Seasonality 6.9%, Soil 5.1%, Yedoma/ground ice 1.1%. Honest multi-factor result — no single dominant control; families sharing information cannot be summed.
   - Family ranking (Abrupt-oriented, OOF fold-refit SHAP): Alpine relief 23%, Annual/dry-season temperature 16%, Land Cover 12%, Thermal continentality 9%, … Fire history **last** (<1%, informative null).
   - **Direction dropped from Fig 7:** nearly all families lean abrupt only because the sample is 94% abrupt (prevalence, not a per-feature finding); the informative direction is *where SHAP crosses zero*, shown per-feature in Fig 8. A caption sentence states the abrupt-orientation + prevalence lean.
   - Domain palette + feature→domain map live in `output/shap_domains.py` (single source of truth, shared with Fig 10).

8. **SHAP mechanism — continuous dependence** — *L6b (deepened)* · new · script `output/fig08_shap_mechanism.py` (see `output/fig08_redesign_spec.md`)
   - **3×3 grid of own-SHAP dependence plots** (each feature's own SHAP vs its own value) for the **top-9 individual indicators** by own `mean|SHAP|` (4 relief / 2 snow / 3 climate — Slope, Annual Mean Temperature, Trend in SWE, Isothermality, Mean Annual SWE, HAND, Trend in precipitation, Mean curvature 500 m, Upstream Area). Per-feature (not family-sum-vs-one-member), so single-feature threshold statements are honest.
   - Shapes reported as **fact**, mechanism reserved for §5.2: Slope's two-regime shape (flat lowland *and* steep alpine favor abrupt; the flat-0° mass peeled out + flagged as sampling-biased), Temperature's warm-edge cliff (> −4 °C → non-abrupt, ~5% boundary cells).

9. **SHAP mechanism — Land Cover per class** — *L6b* · new (split from Fig 8)
   - The categorical analog of the dependence shapes: per-class box of the Land Cover family SHAP (classes with n ≥ 100), signed-sorted, colored by sign. Open Water strongly favors Abrupt; Sedge/Herbaceous favors Non-abrupt. Land Cover is the one family that discriminates in **both** directions (its family SHAP straddles zero) — split off here because a one-hot has no continuous shape.

10. **SHAP dominant-domain map** — *L6c / L7* · new · script `output/fig10_shap_dominance.py` (see `output/fig10_spec.md`)
   - Per-cell TreeSHAP over ~2.85M in-AOA cells (`models/model.json`, all-data), mapped as the **dominant thematic domain** per cell (largest `|Σ domain SHAP|`; **hue only**) + an **area-fraction inset** (% of in-AOA area each domain dominates). Same domain colors as Fig 7 (`shap_domains.py`). Descriptive only, AOA-restricted display.
   - Doubles as the proxy rebuttal — regionally varying drivers ≠ one smooth spatial trend. **Validation gate:** if one domain dominates > ~60% of in-AOA area the map is near-monochrome and the framing is revisited before finalizing.

## Downstream (L7)

11. **Ecoregion breakdown** — *L7* · new
    - Abrupt-favoring fraction by physiographic ecoregion (Alaska Unified Ecoregions); drop ecoregions below a permafrost-coverage threshold; report per-region AOA coverage; in-AOA only.

12. **Olefeldt incumbent contrast** — *L7* · new, **GATED**
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
