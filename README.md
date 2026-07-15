# Using Machine Learning Models to Identify Indicators of Abrupt Thaw

A machine learning (ML) project focused on identifying indicators of abrupt permafrost thaw using geospatial features extracted from remote sensing, community data products, and climate model reanalysis.

## Overview

This repository contains tools and models for:
- Post-processing the Thaw Database to prepare it for ML applications 
- Compiling and extracting geospatial features from Google Earth Engine
- Training and evaluating multiple different ML classification models
- Analyzing model interpretability using Shapley (SHAP) values
- Generating a statewide abrupt-thaw susceptibility (log-evidence) surface from a gridded feature datacube

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

- Python ≥3.13
- Poetry (install from [python-poetry.org](https://python-poetry.org/docs/#installation))

### Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:ethan-pierce/abrupt-thaw-indicators.git
   cd abrupt-thaw-indicators
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

Alternatively, the pyproject.toml file contains all of the information required to install the project and its dependencies using any other package management tool. The poetry.lock file contains detailed package versions to ensure reproducibility.

### Google Earth Engine Authentication

Some scripts require Google Earth Engine authentication. To set this up:

1. Install and authenticate the Earth Engine API:
   ```python
   import ee
   ee.Authenticate()
   ee.Initialize(project='abrupt-thaw-indicators')  # see settings.EE_PROJECT
   ```

2. Or run the authentication command from the command line:
   ```bash
   earthengine authenticate
   ```

## Repository Structure

```
abrupt-thaw-indicators/
├── data/                              # Data processing, feature extraction, prediction I/O
│   ├── Alaska_Permafrost_Thaw_Database_v2.0.0.csv        # ThawDatabase source labels (webb2026-thawdb; used by build_feature_table.py)
│   ├── Database_final_v1.csv          # Alternate database (not used by the current pipeline)
│   ├── build_feature_table.py         # Extract features from Google Earth Engine   ->  features_dirty.csv
│   ├── clean_feature_table.py         # Clean/encode the feature table               ->  features_clean.csv
│   ├── features_dirty.csv             # Raw extracted feature table (build output)
│   ├── features_clean.csv             # Cleaned feature table (input to training/interpretation)
│   ├── build_prediction_data.py       # Build statewide datacube over roi.geojson (1 km)          -> prediction_data.nc
│   ├── roi.geojson                    # Main statewide region of interest
│   └── *.nc                           # Prediction datacubes & outputs (gitignored)
│
├── models/                            # Model training, prediction, interpretation
│   ├── train_xgboost.py               # Train XGBoost (grid-search CV)                -> model.json
│   ├── predict.py                     # Score datacube -> statewide log-evidence susceptibility surface
│   ├── shap_values.py                 # SHAP interpretation of model.json
│   └── model.json                     # Operative trained model (loaded by predict/shap)
│
├── output/                            # Generated figures, maps, and result artifacts
│   ├── archive/                       # Historical figures
│   └── *.png / *.pptx                 # Evaluation figures, SHAP plots, prediction maps, slides
│
├── archive/                           # Legacy code, data, and superseded models
│   ├── train_xgboost_previous_thawdb.py   # Legacy training on an older database
│   ├── model_previous_best.json           # Legacy model (older 53-feature set)
│   ├── model_archive.json                 # Legacy model (old Abrupt=1 encoding era)
│   ├── keras-neural-network.py            # Alternative neural-network approach
│   └── data/, n500/, output/              # Older data/scripts and SHAP outputs
│
├── settings.py                        # Path config (ROOT/DATA/MODELS/OUTPUT; imported by pipeline scripts)
├── CLAUDE.md, MAP.md, SCOPE.md         # Project backbone, location index, manuscript scope
├── pyproject.toml                     # Poetry dependencies and project metadata
└── README.md                          # This file
```

## Usage

### Data Processing Pipeline

1. **Build feature table** (requires Google Earth Engine):
   ```bash
   python data/build_feature_table.py
   ```
   This extracts geospatial features (elevation, slope, land cover, climate variables, etc.) from Google Earth Engine for all points in the thaw database.

2. **Clean feature table**:
   ```bash
   python data/clean_feature_table.py
   ```
   This script removes unnecessary columns, handles missing values, encodes categorical variables, and prepares the data for machine learning.

### Model Training

Train the XGBoost model with cross-validation:
```bash
python models/train_xgboost.py
```

This script:
- Splits data into training/test sets
- Performs grid search cross-validation to optimize hyperparameters
- Evaluates the best model on the test set
- Generates visualizations (confusion matrix, precision-recall curve, feature importance)
- Saves the trained model to `models/model.json`

### Model Interpretation

Generate SHAP values for model interpretability:
```bash
python models/shap_values.py
```

This creates SHAP plots to understand which features are most important for predictions.

### Statewide Prediction

1. **Build the prediction datacube** (requires Google Earth Engine):
   ```bash
   python data/build_prediction_data.py
   ```
   This rasterizes all model features over the region of interest (`data/roi.geojson`, 1 km; terrain served natively per T37) and writes `data/prediction_data.nc`.

2. **Generate maps**:
   ```bash
   python models/predict.py
   ```
   This scores the datacube with `models/model.json` and writes the statewide log-evidence susceptibility map (plus a diagnostic probability map) to `output/`, and the susceptibility/probability NetCDFs to `data/`. No discrete classification is produced (G19/T22).

> **Note:** `models/model.json` is the operative model, regenerated by the rebuild via `train_xgboost.py`. The legacy calibrated variant (`models/model_calibrated.pkl`) is retired and not regenerated; the old calibrated-vs-uncalibrated "canonical model" comparison has been dropped as mooted by the rebuild (see To-Do item 11).

## To-Do List

Manuscript-prep task list, built during a `/grill-with-docs` session (2026-07-08).
Buckets: **Code to modify**, **New analyses needed**, **Manuscript-ready**, **Open
questions**, **Repo hygiene**. Dependency-ordered within each. Scientific calls are
*routed* to science skills, not resolved here (repo convention).

Legend: 🔴 blocking · 🟡 needed for a headline claim · ⚪ nice-to-have.

### Code to modify

**Blocking — pipeline won't run / invalidates current results**

1. ✅ **DONE — Fixed `settings.py` path types.** `ROOT`/`DATA`/`MODELS`/`OUTPUT`
   now use `pathlib.Path` (`settings.py`), so `DATA / 'file.csv'` and `OUTPUT.mkdir()`
   work. Previously `os.path.abspath(...)` returned `str` → `str / str` `TypeError`;
   nothing ran as committed. `archive/` scripts using `os.path.join(DATA, ...)` are
   unaffected (`os.path.join` accepts `Path`). Verified imports resolve.
2. ✅ **DONE (premise corrected) — `Longitude`/`Latitude` were already dropped.**
   The earlier "confirmed retained today" was wrong: the dedup step reassigned
   `feats = todrop.drop_duplicates(...)`, where `todrop` already had lat/lon removed —
   so neither `features_clean.csv` nor the operative `model.json` ever contained
   coordinates (verified by inspecting both). The commented "optional" drop at the
   old lines 109–110 was a red herring. `clean_feature_table.py` now drops lat/lon
   explicitly and unconditionally (legibility only — identical behavior).
   **Consequence:** lat/lon is NOT the source of the AUC-ROC≈0.99 / AUC-PR≈0.9999
   discrimination, and Headline C's SHAP indicators were never raw location — see item 10.
3. ✅ **DONE — Restored the full feature set for retrain #1.** Removed the ad-hoc
   exclusion batches in `clean_feature_table.py` (the "Test: improve interpretability"
   and "remove obvious candidates" drops, plus an earlier bioclim batch) — cut for
   past project-update readability, not on principle. Only the NaN one-hot columns
   are still dropped (structural). Dry-run on the current `features_dirty.csv`
   confirms the clean feature set goes 49 → 69 (+20) with no errors. Retrain #1 uses
   *everything* except lat/lon; the pared set is derived rigorously afterward (item 15).
4. ✅ **DONE — Expanded the prediction datacube to match.** `build_prediction_data.py`
   is feature-name-driven (builds a layer only `if name in feature_names`, stacks in
   model order; `predict.py:55` checks exact match). Closed three gaps that would have
   `KeyError`'d on the full model: (a) `bioclim_vars` extended from 5 → all 19 bands;
   (b) added loader blocks for `Trend in temperature` / `Trend in precipitation`
   (assets `temp-trend` / `annual-precip-trend`, band `scale`); (c) added Land Cover
   codes 73 (Lichens) / 74 (Moss). Lookup dicts are now complete **supersets**, so the
   `if name in feature_names` guards keep the datacube in lockstep automatically for
   both retrains. Static check confirms all 69 full-set features have a loader path.
5. ✅ **DONE — Updated to Thaw Database v2.0.0.** `build_feature_table.py:19`
   now reads `Alaska_Permafrost_Thaw_Database_v2.0.0.csv` (`webb2026-thawdb`,
   `REFERENCES.md`). Schema verified fully compatible with v1.0.0-alpha: identical
   12 columns/order, plain ASCII (existing `latin1` read unchanged), `ThawType`
   categories `Abrupt`/`Non-abrupt` unchanged. 19,540 rows, 18,213 Abrupt / 1,327
   Non-abrupt (93.21% / 6.79%) — exactly matches the published figures. No
   adaptation of `build_feature_table.py` / `clean_feature_table.py` needed.
   Regenerates `features_clean.csv` on the next retrain (#6).

**The retrain (single combined run — #2–#5 all rewrite `features_clean.csv`)**

6. 🔴 Re-run the full pipeline in order:
   `build_feature_table.py` (GEE; needs `ee.Authenticate()` + project
   `ee-abrupt-thaw`) → `clean_feature_table.py` → `train_xgboost.py` (new
   `model.json`) → `build_prediction_data.py` → `predict.py` → `shap_values.py`.

**Headline-map domain**

7. 🟡 **Mask the statewide map to mapped permafrost extent.** `predict.py`
   currently keeps pixels by feature-validity only (`predict.py:94-96`), not by
   any permafrost layer, so it scores mode over non-permafrost ground where the
   target is undefined and the model extrapolates. Obtain the permafrost-extent
   geometry (user will source it), wire a mask into `predict.py`. Not baked in
   anywhere today. Supersedes the old "keep map unmasked so extent can serve as
   post-hoc validation" plan — see item 14.

**Scope reduction**

8. ✅ **Archive the training-lands path entirely.** `build_prediction_data_traininglands.py`,
   `predict_traininglands.py`, and the `*_traininglands.png` / `*_traininglands.nc`
   outputs were built for a past PM update, not the manuscript. **Done 2026-07-13**
   (moved to `archive/`; see TASKS.md T26). Headline map = the statewide 1 km
   **continuous log-evidence susceptibility** surface (`predict.py`); the discrete
   classification map has since been removed entirely (G19/T22).

**Correctness — fix before results are trusted (after retrain)**

9. 🟡 **SHAP split-seed mismatch.** `shap_values.py` re-splits with
   `np.random.default_rng(100)` while `train_xgboost.py` uses seed `42`, so SHAP
   is computed on rows that were in the model's *training* set. Persist the
   trainer's train/test indices (or share the seed) and reuse them. → `/verify-code`

**Methods cleanup (grill-with-docs 2026-07-09) — must land BEFORE the #6 rerun.**
End-to-end methods pass; settled decisions below. Sections C–H still in progress.
Item #9 is subsumed by B6 (canonical config + one operative model).

*A. Data cleaning (`build_feature_table.py` / `clean_feature_table.py`)*
- **A1 — Fire encoding. → SUPERSEDED by T36.** The `Fire Detected` + real-or-NaN
  `Maximum Fire Temperature` scheme was later dropped entirely: FIRMS `T21` is peak
  intensity of one detection (not a regime) and the binary indicator is near-constant
  at 1 km. Fire is now a MODIS MCD64A1 **history** — `Time Since Last Fire` +
  `Burn Count` (right-censored to the ~24-yr record) — plus the retained ALFRESCO
  flammability. See TASKS T36 / `gee_features.py`.
- **A2 — Soil NaN: no change.** Leave pass-through for XGBoost's native routing
  (median-impute for the baseline only). *No* soil-missing indicator — it would fragment
  the lake signal already carried by Land Cover (Open Water) = 42% of points. Measured:
  soil-NaN is 19.4% Abrupt vs 3.4% Non-abrupt (a lake-masking proxy). Lake confound
  consolidated onto Land Cover for the deferred interpretation phase.
- **A3 — Keep exact `drop_duplicates`.** Measured harmless: 1.89% of rows, all
  co-located (≤174 m), all same-label. But the dedup **subset must exclude lat/lon**
  (else co-located twins stop collapsing). Near-duplicate leakage is handled by the CV
  buffer (B5c), not by dedup.

*B. Spatial CV (`spatial_cv.py`, `train_xgboost.py`)*
- **B4 — Nested spatial CV replaces the random split.** Outer folds = headline
  (pooled-OOF AUC-PR + across-fold spread); inner folds = hyperparameter selection.
  Prevents selection/report double-dipping.
- **B5a — Sweep block size** (interpolation→extrapolation); emit AUC-PR + spread at each
  scale (the full curve). How to *quote* it = deferred to the reporting phase.
- **B5b — Equal-area km-grid blocks.** Add an Alaska Albers (EPSG:3338) km-grid block
  method to `spatial_cv.py`; sweep cell **edge length in km** (real, interpretable
  scale). Buffer stays haversine great-circle. Report per-fold Non-abrupt counts so
  minority sparsity at large blocks is visible.
- **B5c — Fixed 1 km buffer** across all block sizes (tied to the ~1 km coarsest
  feature resolution; covers the 62%-within-1 km near-duplicate mass). Block size, not
  buffer, carries broad-scale autocorrelation — avoids a 2-D buffer×block sweep.
- **B6 — Plumbing & reproducibility.** Carry `Latitude`/`Longitude` as non-model columns
  in `features_clean.csv` (spatial CV needs them); **quarantine** them — drop from the
  model matrix with a hard assertion they're absent from `X`. Persist the CV
  **config + seed** (block method, swept sizes, `n_splits`, `buffer_km`, seed) for
  deterministic fold regeneration — not saved index arrays. One operative `model.json`
  = refit on all data with the selected hyperparameters; SHAP + `predict.py` load it
  (SHAP-on-OOF vs. final model = F14/F15).

*C. Training & selection (`train_xgboost.py`)*
- **C8 — Select on pooled-OOF AUC-PR** (average precision, positive = Non-abrupt), matching
  the headline; drop Brier/F1 from selection. (Settled with E13.)
- **C9 — `scale_pos_weight = 1`** (no reweighting). The product is a likelihood-ratio
  index, not a 0.5-thresholded classifier, so reweighting buys nothing and keeps the
  divided-out prior exactly the sample prevalence. (Settled with E13.)
- **C10 — Widen the grid** on principled axes (`max_depth`, `min_child_weight`,
  `reg_lambda`, `learning_rate`, `n_estimators`), small enough for nested CV. Replace the
  opaque `rng.integers()` draws with explicit named seeds
  (`SPLIT_SEED`/`MODEL_SEED`/`CV_SEED`) persisted in the B6 config (42 lineage).

*D. Evaluation & baselines*
- **D11 — Headline = AUC-PR (Non-abrupt)** + prevalence floor (~0.068) + across-fold spread
  (from the B5 sweep). Keep AUC-ROC as a *secondary* (imbalance-insensitive; comparison
  to prior work). **Cut accuracy entirely** (meaningless at 93% prevalence).
- **D12 — Baselines as internal diagnostics.** Dummy (prior/stratified) + penalized
  logistic (sklearn Pipeline: median-impute + standardize) through the *identical* nested
  spatial CV. Purpose is diagnostic — trivial/linear separability ⇒ proxy smell (R#10) —
  NOT a headline "beats logistic" comparison (logistic isn't the incumbent). Depth-1
  stump optional.

*E. Susceptibility index / probability meaning (`predict.py`, calibration)*
- **E13 — Absolute susceptibility = log-evidence / likelihood ratio.** Pipeline product is
  `log-evidence(x) = logit(P_model(abrupt|x)) − logit(π_sample)`, with
  π_sample(abrupt) ≈ 0.932: **absolute, prior-free**, `0 = neutral`, `>0` favors abrupt.
  Explicitly *not* a calibrated probability (sample-prior calibration is indefensible;
  landscape prior is unrecoverable) and *not* a ranking/percentile. It's a post-hoc
  transform on `predict.py` output (subtract the base-rate logit); unbounded — cosmetic
  squash only, disclosed. **Demote** the calibration analysis in `train_xgboost.py` from a
  validity claim to (at most) a monotonicity diagnostic. Prevalence-correction to an
  assumed landscape prior is available *only* as an optional transparent, user-adjustable
  overlay — never baked in. Residual sampling-design (feature-bias) caveat = the deferred
  mechanism-vs-proxy question.

*F. Interpretation (`shap_values.py`)*
- **F14 — Canonical plumbing.** SHAP uses the B6 canonical coordinates + config; remove the
  independent `default_rng(100)` re-split (the old "SHAP on ~70% training rows" bug).
- **F15 — Pooled out-of-fold SHAP (Package 1).** Per outer spatial fold, refit with the
  selected hyperparameters held fixed, compute TreeSHAP on that fold's *held-out* points,
  pool across folds → attributions on genuinely unseen data. Pre-empts the
  SHAP-on-training-data critique: a memorization/proxy-driven feature that doesn't survive
  held-out folds is exposed rather than written into the story. Model compute is negligible
  vs the ~12 h GEE feature build, so cost is a non-issue. The operative refit-on-all model
  still makes the map.

*G. Prediction / mapping (`predict.py`)*
- **G17 — Domain mask = Obu 2019 permafrost probability (soft).** Mask/weight the
  statewide surface by Obu PerProb (`data/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH/`),
  resampled to the 1 km Albers grid — threshold-with-sensitivity or carry P(permafrost)
  as a confidence layer. Replaces the feature-validity-only keep at `predict.py:94-96`
  and supersedes the in-repo Brown 1997 map (`arctic-permafrost-map/`, kept only if a
  categorical cross-check is wanted). Verify the A2 soil-NaN train/serve parity here.
- **G18 — Area-of-Applicability mask.** Add an importance-weighted dissimilarity-to-
  training mask (Meyer & Pebesma 2021) flagging where the grid extrapolates beyond the
  training feature distribution. A late-paper *caveat* layer, not a headline. New code,
  no heavy dependency.
- **G19 — Ditch discrete classification entirely.** The sole product is the continuous
  **log-evidence susceptibility surface** (E13), masked by G17 + G18. Remove the discrete
  classification map / `prediction_classes.nc` from the core pipeline (a classifier-era
  artifact). Any binary "hotspot" map is a deferred *reporting* choice, shown with
  threshold sensitivity if ever done.
- **Glossary + SCOPE.md updated** (2026-07-10) to the log-evidence-index framing:
  `abrupt-thaw susceptibility` is now defined as the likelihood-ratio index, not a
  calibrated probability; SCOPE Headline A / objective 4 / brainstorm aligned.

*H. Reproducibility / closeout*
- **H20.1 — Run manifest.** Persist beside `model.json`: git SHA, `features_clean.csv`
  hash, CV config + seeds, Obu/Brown product versions, selected hyperparameters.
- **H20.2 — Re-run + update `/verify-ml`.** Regenerate `diagnostics/FINDINGS.md` against
  the new pipeline, checking the *new* invariants: coords quarantined from `X`, buffer
  removes near-twins across folds, OOF SHAP explains held-out rows, shuffle-label still
  collapses, baselines run through the same folds.
- **Provenance doc → `PIPELINE.md`** (data sources, script→artifact order, final
  outputs); registered in `MAP.md`. A few custom GEE assets (SWE, climate trends,
  curvature) need their upstream source/DOI confirmed for the methods table.

*Parked / deferred:* minority-sparsity mitigation incl. presence–background → `/ideate`
(separate brief); SHAP mechanism-vs-spatial-proxy → deferred interpretation phase.

### New analyses needed

10. 🟡 **Hunt the source of the near-perfect discrimination.** Reframed: lat/lon was
    never in the model (item 2), so the AUC-ROC≈0.99 / AUC-PR≈0.9999 is *unexplained*,
    not coordinate leakage. After retrain #1, if AUC stays ≈0.99 investigate the real
    cause — near-duplicate lake-cluster points surviving dedup, a feature that proxies
    the label, or genuine separability under 93/7 prevalence. Gates whether *any*
    headline number is manuscript-ready. → `/verify-code`
11. ✅ **REMOVED — canonical-model choice mooted by the rebuild.** The old
    calibrated-vs-uncalibrated decision compared two now-stale artifacts; the
    rebuild produces a single operative `model.json` and the calibrated track
    (`model_calibrated.pkl`) is retired, not regenerated. Do not re-raise. (The
    prior "37% of pixels flip" figure was stale and was never lat/lon leakage —
    see item 2.)
12. 🟡 **Threshold-sensitivity analysis (not threshold *selection*).** Present the
    continuous surface as primary; show the discrete map at several thresholds as
    a sensitivity band rather than claiming one "correct" cut. Rationale under
    Open questions. → `/analyze-system`
13. 🟡 **Sampling-bias scoping of the SHAP story (Headline C) — DEFERRED to the
    interpretation phase.** Does the explanation encode thaw *mechanism* or merely
    lake-proximity, given `webb2026-thawdb`'s lake dominance (10,625/18,213 abrupt
    points are lakes)? This is a *post-retrain-#2, results-in-hand* question, not a
    rebuild task — do not re-litigate it during the pipeline rebuild; revisit only
    once there are SHAP results to interpret. → `/analyze-system`
14. 🟡 **Positioning comparison = the real "validation" (objective 5).** Compare
    the mode surface against thaw-*character* products, NOT permafrost extent:
    Olefeldt categorical thermokarst classes and occurrence-susceptibility
    (`wang2023-arctic-tls`). Superseded plan — permafrost extent as post-hoc
    validation — dropped: it validates presence not mode, and the reference
    ground-ice products are known-coarse (`webb2026-thawdb`: 65% miss).
    **Inversion worth pursuing:** frame the fine-scale surface as *revealing*
    structure the coarse ground-ice/extent maps miss — offense for objective 5,
    not defensive validation. → `/analyze-system`
15. 🟡 **Derive a rigorous feature-paring protocol → retrain #2.** After retrain
    #1 (full set), pare on stated grounds — collinearity (VIF), coverage/quality,
    leakage — NOT interpretability aesthetics. Document one rule per exclusion in
    methods; this converts the old ad-hoc cuts into a defensible protocol and
    inoculates Headline C against the "you curated features to get the story"
    critique. If a dropped variable is a plausible driver, escalate rather than
    silently cut. Produces the final model. → `/analyze-system` + `/verify-code`

### Manuscript-ready (no further tinkering)

Blunt truth: the two retrains regenerate every model number, figure, map, and
SHAP plot, so **no quantitative result is final yet.** What is stable and
citable now is the scaffolding, not the results:

- **Data foundation** — the Thaw Database, published/peer-reviewed as
  `webb2026-thawdb` (once the pipeline is on v2.0.0, item 5).
- **Literature positioning** — `SCOPE.md` Key background + `REFERENCES.md`
  (grounded during the session): the stage/occurrence/**mode** framing, and the
  honest novelty (mode + explanation, not continuous-vs-categorical).
- **Method narrative** — XGBoost + SHAP applied to labeled points and a statewide
  feature stack; the *approach* is stable even though every number it produces
  will be regenerated.
- **Feature-extraction design** — the GEE sampling methodology in
  `build_feature_table.py` (reducers, scales, sources), modulo the datacube
  expansion (item 4).

Everything else (metrics, calibration curves, confusion matrices, the maps, the
SHAP rankings) is provisional until after retrain #2.

### Open questions (carry forward)

- **Fire representation adequacy — RESOLVED (T36).** Settled on fire *history*:
  dropped FIRMS max-fire-temp / `Fire Detected`, added MODIS MCD64A1
  `Time Since Last Fire` + `Burn Count` (right-censored ~24 yr; NaN above ~70°N).
  See `SCOPE.md`.
- **DB v2.0.0 schema compatibility — RESOLVED (item 5).** v2.0.0 CSV supplied and
  diffed against v1.0.0-alpha: identical columns/order/encoding and `ThawType`
  categories; no code adaptation needed.

### Repo hygiene / documentation (⚪ nice-to-have)

- [ ] Formalize and document feature exclusion protocol (folds into item 15)
- [ ] Create evaluation metrics dashboard
- [ ] Document Google Earth Engine asset requirements
- [ ] Document feature engineering pipeline and feature definitions
- [ ] Add example prediction scripts for new data
- [ ] Create comprehensive API documentation (optional)
- [ ] Add CI/CD workflows (optional)

## Dependencies

Key dependencies include:
- `xgboost` - Gradient boosting framework
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `earthengine-api` - Google Earth Engine integration
- `shap` - Model interpretability
- `matplotlib`, `seaborn` - Visualization
- `keras`, `jax` - Deep learning (for alternative approaches)
- `imbalanced-learn` - Handling class imbalance

See `pyproject.toml` for complete dependency list and versions.

## License

This project is licensed under the GPL-3.0-or-later License.

## Authors

- **Ethan Pierce** - ethan.g.pierce@dartmouth.edu

## Contributing

This is an active research project. For questions or contributions, please contact the project maintainers.
