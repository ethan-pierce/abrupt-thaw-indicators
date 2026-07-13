# TASKS — methods-cleanup implementation

Atomic, dependency-ordered task list for the methods-cleanup rewrite. Rationale
for each lives in `README.md` → "Methods cleanup (grill-with-docs 2026-07-09/10)"
(cited as [A1]…[H20]); target pipeline in `PIPELINE.md`. An agent takes the next
box whose *Depends* are all checked. Line numbers are current-code anchors — verify
before editing.

**PRIORITY OVERRIDE:** T30 (below) supersedes the normal dependency ordering — it is
the top priority and must be resolved before any other task is taken. (T0 is done.)

## PRIORITY — do before any other task

- [ ] **T30 — Heavy inline-GEE features hang at full-N point sampling.**
  [ee-project-access-lost] *Depends:* T0 *Done when:* Maximum Fire Temperature,
  Mean Annual SWE, and Trend in SWE/precip/temperature each resolve over all 19,540
  points in `build_feature_table.py` to completion with correct values in minutes
  (not tens of minutes), without the `try/except` swallowing a timeout; parity with
  the datacube path preserved; no custom asset reintroduced.

  **Problem:** T0 replaced precomputed assets with on-the-fly GEE computations
  (`data/gee_features.py`). Sampling a deep temporal reduction at 19,540 points via
  the legacy `add_feature` pattern (`points.map(reduceRegion)` + one
  `computeFeatures`) degrades catastrophically. FIRMS max (`ImageCollection('FIRMS')
  .select('T21').max()`, ~9,000 images): datacube single-`sampleRectangle` path
  **completes in 219 s**, but the point path at full N **hung >26 min with no error**
  (killed). SWE + the 3 trends are the same shape of risk but were **not** reached in
  testing — verify them too. Everything else from T0 is verified PASS at full scale.

  **Candidate fixes (unchosen):** chunk points into batches; use
  `reduceRegions`/`sampleRegions` (correct band handling — single-mean → `mean`);
  or compute once over Alaska (datacube-style, which works) and sample points from
  the materialized array. Code: `data/gee_features.py`, `data/build_feature_table.py`
  (`add_feature`), `data/build_prediction_data.py` (working datacube path).

- [x] **T0 — Re-establish the GEE project & custom assets.** [ee-project-access-lost]
  Access to the `ee-abrupt-thaw` project was lost (2026-07-10); its 13 custom feature
  assets had **no local source copies**. Resolved by rebuilding the feature side with
  **zero custom assets** (two tracks). *Depends:* — *Done when:* the feature build
  resolves every feature with **no custom GEE asset** and no `ASSET_ROOT` dependency,
  from public GEE catalog data + local rasters. **✓ 2026-07-13.**

  > **DONE (2026-07-13).** Feature sourcing fully rebuilt with zero custom assets. New
  > modules `data/gee_features.py` (GEE track: TAGEE-family mean curvature from 3DEP;
  > SWE + SWE/precip/temp trends from Daymet V4 `linearFit`; max fire temp from FIRMS)
  > and `data/local_rasters.py` (LOCAL track: nearest point/grid sampling of
  > `data/{alfresco,NLCD2016,snap}` + Obu). `build_feature_table.py` and
  > `build_prediction_data.py` both restructured onto the two tracks; `ASSET_ROOT`
  > **removed from `settings.py`** and from all code (`plot_input_data.py` too). LOCAL
  > track validated offline over all 19,540 ThawDB points (97–100% coverage, plausible
  > ranges). **Not yet executed against GEE** — the GEE track (curvature/Daymet/FIRMS)
  > is import/compile-verified but its first real run is the ~12h feature build, deferred
  > under dev-mode; validate the TAGEE curvature port on GEE before trusting those two
  > columns. Full sourcing map: `PIPELINE.md`. Context: memory [[ee-project-access-lost]].
  1. [x] Create/choose the new EE project; set `EE_PROJECT` in `settings.py`. **Done
     2026-07-13:** new project `abrupt-thaw-indicators` (EE API enabled).
  2. [x] Test-read each old asset — **Done 2026-07-13: all 13 FAILED** (old project
     inaccessible — `earthengine.assets.list` denied), so no shared-ACL shortcut.
  3. [x] Readable assets → keep/copy. N/A — none readable.
  4. [x] **Done 2026-07-13.** Restructured `build_feature_table.py` into two tracks:
     **(GEE)** `gee_features.py` — curvature (3DEP + TAGEE-family `ee` port, Zevenbergen-
     Thorne mean curvature at cell size = window/2), Mean Annual SWE + SWE/precip/temp
     trends (Daymet V4 + `linearFit`), Maximum Fire Temperature (FIRMS `T21`);
     **(LOCAL)** `local_rasters.py` — nearest point-sampling of `data/{snap,alfresco,
     NLCD2016}`, reprojecting each point to the raster CRS (nearest for all: matches the
     original point `reduceRegion(mean)` = covering pixel, and is required for the
     categorical veg-mode/NLCD layers). Reconstructed params documented in each module's
     docstring; still to copy into the methods table (T29 remainder).
  5. [x] **Done 2026-07-13.** Obu2019 domain mask for T20: the **local** PerProb GeoTIFF
     (`data/Obu2019/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif`) samples cleanly with
     rasterio (0–1, 99.7% coverage over training points) — no asset upload. Exposed as
     `local_rasters.OBU_TIF` + `sample_points` for T20 to consume.
  6. [x] **Done 2026-07-13.** Verified: no live `ASSET_ROOT` reference remains in any
     `.py` (grep clean; `ASSET_ROOT` deleted from `settings.py`); both build scripts and
     `plot_input_data.py` `py_compile` clean and resolve every feature from public
     catalog loads + local rasters.

## Stage 0 — CV foundation (`models/spatial_cv.py`) — do first

- [x] **T1 — Add equal-area km-grid block method.** [B5b] Add an Alaska Albers
  (EPSG:3338) grid option to `assign_blocks`: reproject lat/lon → metres, floor to
  cells of a given **edge length in km**. Keep the buffer on haversine. Extend
  `_selftest`. *Depends:* — *Done when:* `assign_blocks(..., method='albers_grid',
  cell_km=k)` returns stable ids and `_selftest` passes; check a projection lib
  (`pyproj`) is available or add it.
- [x] **T2 — Nested-fold helper.** [B4] Add a helper that yields nested folds: outer
  `buffered_block_folds` over all points; for each outer fold, inner
  `buffered_block_folds` over the outer-train subset. *Depends:* T1 *Done when:* a
  self-test confirms inner folds never touch the outer test blocks.
- [x] **T3 — Per-fold minority reporting.** [B5b] Have the fold machinery expose the
  Gradual (class-1) count per fold. *Depends:* T1 *Done when:* counts are
  returned/logged for each fold at each block size.

## Stage 1 — Data cleaning (`data/clean_feature_table.py`)

- [x] **T4 — Fire encoding.** [A1] Remove `Maximum Fire Temperature` from the
  `fillna` list (drop the `→0.0` fill, ~line 66). Add a binary `Fire Detected`
  column = `notna(Maximum Fire Temperature)`. Leave the continuous value real-or-NaN.
  *Depends:* — *Done when:* `features_clean.csv` has `Fire Detected` ∈ {0,1} and the
  temp column retains NaNs.
- [x] **T5 — Dedup excludes coordinates.** [A3/B6] Change `drop_duplicates()`
  (~line 109) to `drop_duplicates(subset=<all feature cols except Latitude/Longitude>)`.
  *Depends:* T6 *Done when:* dup count matches the pre-change exact-feature dedup
  (~369 rows / 105 groups on current data) with coords retained.
- [x] **T6 — Carry coordinates through.** [B6] Remove the unconditional lat/lon drop
  (~line 107); keep `Latitude`/`Longitude` as columns in `features_clean.csv`.
  *Depends:* — *Done when:* `features_clean.csv` includes both coordinate columns.

## Stage 2 — Training & selection (`models/train_xgboost.py`) — largest rewrite

- [x] **T7 — Load + quarantine coords.** [B6] `X = feats.drop(['Class','Latitude',
  'Longitude'])`, `y = feats['Class']`, `coords = feats[['Latitude','Longitude']]`;
  `assert 'Latitude' not in X and 'Longitude' not in X`. *Depends:* T6 *Done when:*
  the assertion is in place and passes.
- [x] **T8 — Nested spatial CV over a block-size sweep.** [B4/B5] Replace
  `train_test_split` + `StratifiedKFold` `GridSearchCV` with the T2 nested folds run
  at each block size in the sweep (interpolation→extrapolation), buffer = 1 km.
  *Depends:* T1,T2,T7 *Done when:* the trainer produces pooled-OOF predictions per
  outer fold at each block size.
- [x] **T9 — Selection metric.** [C8] Inner loop selects on pooled-OOF **AUC-PR**
  (`average_precision_score`, positive = Gradual). *Depends:* T8 *Done when:* selected
  hyperparameters maximise inner pooled-OOF AP; Brier/F1 removed from selection.
- [x] **T10 — `scale_pos_weight = 1`.** [C9] Remove the class-ratio reweighting
  (~line 73). *Depends:* T8 *Done when:* the estimator factory sets no imbalance
  reweighting.
- [x] **T11 — Grid breadth + named seeds.** [C10] Widen the grid on `max_depth`,
  `min_child_weight`, `reg_lambda`, `learning_rate`, `n_estimators`. Replace the
  `rng.integers()` draws with explicit `SPLIT_SEED`/`MODEL_SEED`/`CV_SEED`, and persist
  the CV config + seeds to a file. *Depends:* T8 *Done when:* config+seeds are written
  and rerunning reproduces identical folds.
- [x] **T12 — Headline metrics.** [D11] Emit AUC-PR-vs-block-size curve + across-fold
  spread + prevalence floor (~0.068). Keep AUC-ROC as secondary. **Remove all accuracy
  reporting.** *Depends:* T8,T9 *Done when:* outputs contain the curve and no accuracy.
- [x] **T13 — Baselines as diagnostics.** [D12] Run a dummy (prior/stratified) and a
  penalized logistic (sklearn Pipeline: median-impute + standardize) through the same
  nested folds. Optional depth-1 stump. *Depends:* T8 *Done when:* baseline pooled-OOF
  AUC-PR is reported alongside XGBoost.
- [x] **T14 — Operative model.** [B6] Refit on all data with the selected
  hyperparameters → `models/model.json`. *Depends:* T9,T10,T11 *Done when:*
  `model.json` is the all-data refit and loads in `predict.py`/`shap_values.py`.
- [x] **T15 — Demote calibration.** [E13] (delete-only; calibration block removed in the T8–T12 rewrite, no monotonicity diagnostic re-added per decision 2026-07-10) Reframe the calibration block (ECE/reliability
  curves) as a monotonicity diagnostic, not a validity claim. *Depends:* — *Done when:*
  no calibration metric is presented as headline validity.
- [x] **T16 — Run manifest.** [H20.1] Write beside `model.json`: git SHA,
  `features_clean.csv` hash, CV config+seeds, Obu/Brown versions, selected
  hyperparameters. *Depends:* T11,T14 *Done when:* the manifest file is produced each run.
- [x] **T17 — Remove mislabeled "CV F1".** [C8] Delete the `split0..4` "CV F1" block
  (~lines 213–230). *Depends:* T8 *Done when:* gone.

## Stage 3 — Prediction datacube (`data/build_prediction_data.py`)

- [x] **T18 — Add `Fire Detected` layer.** [A1 parity] Build a `Fire Detected` grid
  layer (from `max-fire-temp` NaN-ness) so the datacube feature set matches the model.
  *Depends:* T4 *Done when:* the datacube contains `Fire Detected` and the
  feature-name match in `predict.py` passes.

## Stage 4 — Mapping (`models/predict.py`)

- [x] **T19 — Log-evidence output.** [E13] Emit
  `log_evidence = logit(P_model(abrupt|x)) − logit(π_sample)` (π_sample(abrupt) ≈ 0.932)
  as the susceptibility surface. *Depends:* T14 *Done when:* the primary raster is
  log-evidence, `0 = neutral`.
- [ ] **T20 — Obu domain mask.** [G17] Soft-mask/weight by Obu PerProb
  (`data/Obu2019/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif`; sample via
  `local_rasters.OBU_TIF` / `sample_points`), resampled to the 4 km Albers grid;
  replaces the feature-validity-only keep (`predict.py:94-96`). *Depends:* T19, T0
  *Done when:* off-permafrost pixels are masked/down-weighted.
- [ ] **T21 — AOA mask.** [G18] Add an importance-weighted dissimilarity-to-training
  mask with a CV-derived threshold; output as a reliability layer. *Depends:* T14,T19
  *Done when:* an extrapolation-flag raster is produced.
- [ ] **T22 — Remove discrete classification.** [G19] Delete the discrete class output
  (`prediction_classes.nc`, classification map). *Depends:* T19 *Done when:* no discrete
  class artifact is written.
- [ ] **T23 — Soil-NaN train/serve parity.** [A2] Verify the datacube reproduces soil
  NaN the same way as the points (so native routing transfers). *Depends:* — *Done when:*
  parity confirmed (documented check).

## Stage 5 — Interpretation (`models/shap_values.py`)

- [x] **T24 — Canonical plumbing.** [F14] Remove the independent `default_rng(100)`
  re-split; load the B6 coords + persisted CV config. *Depends:* T11 *Done when:* no
  independent split remains. **✓ 2026-07-13:** `shap_values.py` rewritten — the
  `train_test_split` is gone; `load_inputs` applies the T7 coord quarantine and
  `load_cv_config` reads `models/cv_config.json` (with trainer-default fallbacks for a
  stale config missing `operative_cell_km`).
- [x] **T25 — Pooled out-of-fold SHAP.** [F15] Per outer fold, refit with the selected
  hyperparameters fixed, TreeSHAP on held-out points, pool across folds. *Depends:*
  T2,T14,T24 *Done when:* SHAP outputs are computed only on held-out rows. **✓
  2026-07-13:** `pooled_oof_shap` refits per single-level buffered block fold (operative
  cell km, fixed hyperparameters from `selected_hparams.json`) and runs TreeSHAP on the
  held-out points only, pooling so each point is explained out-of-fold. Margin/log-odds
  space (`model_output='raw'`, tree_path_dependent), negated to Abrupt (class 0)
  orientation per the T19 log-evidence scale. `model.json` intentionally unused (OOF needs
  per-fold refits). Smoke-verified (`SHAP_SMOKE=1`) end-to-end; **authoritative run
  deferred** until the rebuilt features + trainer run produce `selected_hparams.json` and
  a matching `features_clean.csv` (needs T30).

## Stage 6 — Scope reduction & closeout

- [x] **T26 — Archive training-lands path.** [README #8] Move
  `build_prediction_data_traininglands.py`, `predict_traininglands.py`, and
  `*_traininglands.*` outputs to `archive/`. *Depends:* — *Done when:* moved.
  **✓ 2026-07-13:** `build_prediction_data_traininglands.py` → `archive/data/`,
  `predict_traininglands.py` → `archive/`, both `*_traininglands.png` → `archive/output/`
  (no `.nc` outputs on disk — gitignored/never generated). No live script imports them
  (only README/PIPELINE docs referenced them, now updated). Live tree grep-clean.
- [x] **T27 — Retire calibrated artifacts.** [pipeline-rebuild-v2] Confirm nothing
  regenerates `model_calibrated*`; archive the stale files. *Depends:* — *Done when:*
  no live script references them. **✓ 2026-07-13:** verified only generator was
  `train_xgboost_calibrated.py` itself (nothing else consumes `model_calibrated*`;
  `predict.py`'s "calibrated" mentions are descriptive text only). Archived to `archive/`:
  `train_xgboost_calibrated.py`, `model_calibrated.pkl`, `model_calibrated_base.json`, and
  all `output/*_calibrated.png`. Also swept two orphaned stale curves
  (`calibration_curve.png`, `calibration_curve_enhanced.png`) — no generator since T15.
- [ ] **T28 — Update `/verify-ml` + regenerate FINDINGS.** [H20.2] Re-point the
  `diagnostics/` suite at the new invariants (coord quarantine, buffer removal, OOF
  SHAP on held-out, baselines) and regenerate `FINDINGS.md`. *Depends:* T14,T19,T25
  *Done when:* `FINDINGS.md` is stamped to the new pipeline.

## Non-code (user / methods table)

- [x] **T29 — Acquire the LOCAL-track source rasters.** **Done 2026-07-13.** GEE-track
  upstreams resolved (Daymet V4, FIRMS, 3DEP+TAGEE — public, no acquisition). SNAP
  "projected" ×3: UAF SNAP AR5/CMIP5 771 m, 5modelAvg/RCP 8.5, change = 2090–99 − 2010–19
  (JJA/DJF/annual) → `data/fetch_snap_projections.py` → `data/snap/`. ALFRESCO ×2:
  historical CRU/observed runs → `data/fetch_alfresco.py` → `data/alfresco/` (flammability
  continuous; vegetation mode categorical 0–8). NLCD-2016 Alaska: user-provided ERDAS
  `.img`+`.ige` in `data/NLCD2016/`. All EPSG:3338/WGS84-Albers, rasterio-readable,
  smoke-tested. *Remaining (non-blocking):* record the reconstructed GEE-track params
  (curvature smoothing / Daymet year-range / aggregation) in the methods table.
