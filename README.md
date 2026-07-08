# Using Machine Learning Models to Identify Indicators of Abrupt Thaw

A machine learning (ML) project focused on identifying indicators of abrupt permafrost thaw using geospatial features extracted from remote sensing, community data products, and climate model reanalysis.

## Overview

This repository contains tools and models for:
- Post-processing the Thaw Database to prepare it for ML applications 
- Compiling and extracting geospatial features from Google Earth Engine
- Training and evaluating multiple different ML classification models
- Analyzing model interpretability using Shapley (SHAP) values
- Generating statewide abrupt-thaw probability and classification maps from a gridded feature datacube

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
   ee.Initialize(project='ee-abrupt-thaw')
   ```

2. Or run the authentication command from the command line:
   ```bash
   earthengine authenticate
   ```

## Repository Structure

```
abrupt-thaw-indicators/
├── data/                              # Data processing, feature extraction, prediction I/O
│   ├── Alaska_Permafrost_Thaw_Database_v1.0.0-alpha.csv  # ThawDatabase source labels (used by build_feature_table.py)
│   ├── Database_final_v1.csv          # Alternate database (not used by the current pipeline)
│   ├── build_feature_table.py         # Extract features from Google Earth Engine   ->  features_dirty.csv
│   ├── clean_feature_table.py         # Clean/encode the feature table               ->  features_clean.csv
│   ├── features_dirty.csv             # Raw extracted feature table (build output)
│   ├── features_clean.csv             # Cleaned feature table (input to training/interpretation)
│   ├── build_prediction_data.py       # Build statewide datacube over roi.geojson (4 km)          -> prediction_data.nc
│   ├── build_prediction_data_traininglands.py  # Datacube over training-lands.geojson (500 m)     -> prediction_data_traininglands.nc
│   ├── roi.geojson                    # Main statewide region of interest
│   ├── training-lands.geojson         # Alternate "training lands" region of interest
│   └── *.nc                           # Prediction datacubes & outputs (gitignored)
│
├── models/                            # Model training, prediction, interpretation
│   ├── train_xgboost.py               # Train XGBoost (grid-search CV)                -> model.json
│   ├── train_xgboost_calibrated.py    # Calibrated variant (early stopping + sigmoid) -> model_calibrated.pkl / _base.json
│   ├── predict.py                     # Score datacube -> statewide probability & classification maps
│   ├── predict_traininglands.py       # Same, for the training-lands datacube
│   ├── shap_values.py                 # SHAP interpretation of model.json
│   ├── model.json                     # Operative trained model (loaded by predict/shap)
│   ├── model_calibrated.pkl           # Calibrated model — under evaluation (see SCOPE.md)
│   └── model_calibrated_base.json     # Base model inside the calibrated wrapper
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
   This rasterizes all model features over the region of interest (`data/roi.geojson`, ~4 km) and writes `data/prediction_data.nc`. A finer-resolution variant over `data/training-lands.geojson` (~500 m) is available via `data/build_prediction_data_traininglands.py`.

2. **Generate maps**:
   ```bash
   python models/predict.py
   ```
   This scores the datacube with `models/model.json` and writes statewide probability and classification maps to `output/`, plus prediction NetCDFs to `data/`. Use `models/predict_traininglands.py` for the training-lands datacube.

> **Note:** `models/model.json` is the operative model. A calibrated variant (`models/model_calibrated.pkl`) also exists; the choice between them, and the robustness of the statewide map, are open scientific questions (see the To-Do List below, item 11). An earlier comparison found they change ~37% of map pixels, but that figure predates the lat/lon leakage fix and is being re-measured.

## To-Do List

Manuscript-prep task list, built during a `/grill-with-docs` session (2026-07-08).
Buckets: **Code to modify**, **New analyses needed**, **Manuscript-ready**, **Open
questions**, **Repo hygiene**. Dependency-ordered within each. Scientific calls are
*routed* to science skills, not resolved here (repo convention).

Legend: 🔴 blocking · 🟡 needed for a headline claim · ⚪ nice-to-have.

### Code to modify

**Blocking — pipeline won't run / invalidates current results**

1. 🔴 **Fix `settings.py` path types.** `DATA`/`MODELS`/`OUTPUT` (`settings.py:4-6`)
   are plain strings via `os.path.abspath(...)`, but every consumer uses
   `pathlib`-style `DATA / 'file.csv'` (e.g. `clean_feature_table.py:137`,
   `predict.py:244` calls `OUTPUT.mkdir`). `str / str` → `TypeError`; nothing
   runs as committed. Wrap the three in `pathlib.Path(...)`. Gates all
   reproducibility. Trivial, no trade-off.
2. 🔴 **Drop `Longitude`/`Latitude` from the feature table.** Uncomment
   `clean_feature_table.py:109-110` so lat/lon are dropped from `feats` (not just
   from the throwaway `todrop` dup-check at 113-114). Confirmed retained today →
   prime suspect for the AUC-ROC≈0.99 / AUC-PR≈0.9999 spatial leakage, and it
   poisons Headline C (SHAP "indicators" become raw location). Forces a retrain.
3. 🔴 **Restore the full feature set for retrain #1.** Remove the ad-hoc
   exclusion batches in `clean_feature_table.py:118-132` (dropped under "Test:
   improve interpretability" / "remove obvious candidates" — originally cut for
   project-update readability, not on principle). Retrain #1 uses *everything*
   except lat/lon; the pared set is derived rigorously afterward (item 15).
4. 🔴 **Expand the prediction datacube to match.** `build_prediction_data.py`
   assembles only a bioclim subset (bio01/04/07/12/15); once the model trains on
   the full set, the datacube must supply every model feature or `predict.py`'s
   feature-name check fails. Keep training features and datacube features in
   lockstep for both retrains.
5. 🔴 **Update to Thaw Database v2.0.0.** Pipeline reads
   `Alaska_Permafrost_Thaw_Database_v1.0.0-alpha.csv` (`build_feature_table.py:19`);
   authoritative published version is v2.0.0 (`webb2026-thawdb`, `REFERENCES.md`).
   Obtain the v2.0.0 CSV, add to `data/`, repoint the read. **Risk:** verify
   column names + `ThawType` categories (and the non-abrupt / negative-sample
   provenance) match v1.0.0-alpha, else adapt `build_feature_table.py` and
   `clean_feature_table.py`. Combines with #2 into one retrain.

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

8. ⚪ **Archive the training-lands path entirely.** `build_prediction_data_traininglands.py`,
   `predict_traininglands.py`, and the `*_traininglands.png` / `*_traininglands.nc`
   outputs were built for a past PM update, not the manuscript. Move to `archive/`.
   Headline map = the statewide 4 km **continuous probability** surface (`predict.py`);
   the discrete classification map is secondary.

**Correctness — fix before results are trusted (after retrain)**

9. 🟡 **SHAP split-seed mismatch.** `shap_values.py` re-splits with
   `np.random.default_rng(100)` while `train_xgboost.py` uses seed `42`, so SHAP
   is computed on rows that were in the model's *training* set. Persist the
   trainer's train/test indices (or share the seed) and reuse them. → `/verify-code`

### New analyses needed

10. 🟡 **Re-check discrimination after the lat/lon drop.** Confirm AUC falls to a
    defensible range; if it stays ≈0.99, hunt the next leakage source
    (feature-independence check). Gates whether *any* headline number is
    manuscript-ready. → `/verify-code`
11. 🟡 **Canonical-model decision (calibrated vs. uncalibrated) — RE-DO.** Prior
    "37% of pixels flip / Pearson r≈0.66 / mean P(Abrupt) 0.47 vs 0.80" was
    measured on lat/lon-leaking models and is now STALE. Re-measure divergence on
    the clean retrained pair, then choose the canonical model. Note the calibrated
    track (`model_calibrated.pkl`) is currently orphaned — no predict/SHAP script
    consumes it. **Base-rate caveat:** the calibrated model was sigmoid-calibrated
    to the DB's ~93%-abrupt prior, which is a *sampling artifact* (lake/road bias),
    not the landscape prevalence — so its probabilities may be calibrated to the
    wrong base rate. → `/analyze-system`
12. 🟡 **Threshold-sensitivity analysis (not threshold *selection*).** Present the
    continuous surface as primary; show the discrete map at several thresholds as
    a sensitivity band rather than claiming one "correct" cut. Rationale under
    Open questions. → `/analyze-system`
13. 🟡 **Sampling-bias scoping of the SHAP story (Headline C).** Does the
    explanation encode thaw *mechanism* or merely lake-proximity, given
    `webb2026-thawdb`'s lake dominance (10,625/18,213 abrupt points are lakes)?
    → `/analyze-system`
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

- **Canonical model: calibrated vs. uncalibrated** — deferred to post-retrain-#1
  re-measurement; leaning uncalibrated on the base-rate argument (item 11), but
  it's a science call → `/analyze-system`.
- **Fire representation adequacy** — fire is present (not missing), but is
  instantaneous max-fire-temp / flammability the right encoding, or is fire
  *history* / time-since-fire needed? → `/analyze-system` (see `SCOPE.md`).
- **DB v2.0.0 schema compatibility** — resolves once the user supplies the
  v2.0.0 CSV and we diff its columns/categories against v1.0.0-alpha (item 5).

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
