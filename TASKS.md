# TASKS — methods-cleanup implementation

Atomic, dependency-ordered task list for the methods-cleanup rewrite. Rationale
for each lives in `README.md` → "Methods cleanup (grill-with-docs 2026-07-09/10)"
(cited as [A1]…[H20]); target pipeline in `PIPELINE.md`. An agent takes the next
box whose *Depends* are all checked. Line numbers are current-code anchors — verify
before editing.

**PRIORITY OVERRIDE:** T0 (below) supersedes the normal dependency ordering — it is
the top priority and must be resolved before any other task is taken.

## PRIORITY — do before any other task

- [ ] **T0 — Re-establish the GEE project & custom assets.** [ee-project-access-lost]
  Access to the `ee-abrupt-thaw` project was lost (2026-07-10); its 13 custom feature
  assets have **no local source copies**, so the entire feature side of the pipeline is
  unreproducible until they are re-established. **Blocks the GEE datacube (T18), the Obu
  mask (T20), and any feature/label rebuild — take it before anything else.**
  *Depends:* — (upstream re-sourcing is gated by T29) *Done when:* a working EE project
  resolves every custom asset, Obu2019 is uploaded for T20, and `settings.py`
  `EE_PROJECT`/`ASSET_ROOT` point at it. Steps:
  1. Create/choose the new EE project; set `EE_PROJECT` in `settings.py`.
  2. Test-read each `{ASSET_ROOT}/…` asset from it — GEE asset ACLs are independent of
     the compute project, so shared/public assets still resolve without owning
     `ee-abrupt-thaw`. Keep `ASSET_ROOT` on the old prefix if they do.
  3. Readable assets → keep in place, or copy into the new project for ownership.
  4. Unreadable assets → re-source from upstream (SWE / SWE-trend / precip-trend /
     temp-trend / projected-climate / curvature upstreams are unconfirmed → T29) and
     re-upload under `ASSET_ROOT`. The 13: `NLCD-2016`, `AK-curvature-500m`,
     `AK-curvature-2k`, `ALFRESCO-historical-flammability`,
     `ALFRESCO-historical-vegetation-mode`, `max-fire-temp`, `ee-mean-annual-swe`,
     `annual-swe-trend`, `annual-precip-trend`, `temp-trend`,
     `summer-temperature-trend`, `winter-temperature-trend`,
     `annual-precipitation-trend`.
  5. Upload Obu2019 PerProb (`data/Obu2019/*.tif`) as an asset for T20.
  6. Verify `build_feature_table.py` and `build_prediction_data.py` resolve every asset.

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
  (`data/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH/`), resampled to the 4 km Albers
  grid; replaces the feature-validity-only keep (`predict.py:94-96`). *Depends:* T19
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

- [ ] **T24 — Canonical plumbing.** [F14] Remove the independent `default_rng(100)`
  re-split; load the B6 coords + persisted CV config. *Depends:* T11 *Done when:* no
  independent split remains.
- [ ] **T25 — Pooled out-of-fold SHAP.** [F15] Per outer fold, refit with the selected
  hyperparameters fixed, TreeSHAP on held-out points, pool across folds. *Depends:*
  T2,T14,T24 *Done when:* SHAP outputs are computed only on held-out rows.

## Stage 6 — Scope reduction & closeout

- [ ] **T26 — Archive training-lands path.** [README #8] Move
  `build_prediction_data_traininglands.py`, `predict_traininglands.py`, and
  `*_traininglands.*` outputs to `archive/`. *Depends:* — *Done when:* moved.
- [ ] **T27 — Retire calibrated artifacts.** [pipeline-rebuild-v2] Confirm nothing
  regenerates `model_calibrated*`; archive the stale files. *Depends:* — *Done when:*
  no live script references them.
- [ ] **T28 — Update `/verify-ml` + regenerate FINDINGS.** [H20.2] Re-point the
  `diagnostics/` suite at the new invariants (coord quarantine, buffer removal, OOF
  SHAP on held-out, baselines) and regenerate `FINDINGS.md`. *Depends:* T14,T19,T25
  *Done when:* `FINDINGS.md` is stamped to the new pipeline.

## Non-code (user / methods table)

- [ ] **T29 — Confirm custom-asset sources.** Fill in upstream product/DOI for the SWE,
  climate-trend, projected-climate, and curvature GEE assets (flagged in `PIPELINE.md`).
