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
│   ├── Alaska_Permafrost_Thaw_Database_v2.0.0.csv        # ThawDatabase source labels (webb2026-thawdb; read by build_feature_table.py)
│   ├── build_feature_table.py         # Extract features (GEE + local rasters)         -> features_dirty.csv
│   ├── clean_feature_table.py         # Clean/encode the feature table                 -> features_clean.csv
│   ├── gee_features.py                # Inline GEE feature computation (public catalog)
│   ├── local_rasters.py               # Point/grid sampling of downloaded source rasters
│   ├── build_daymet_rasters.py        # Materialize Daymet SWE + climate-trend rasters
│   ├── build_modis_fire_rasters.py    # Materialize MODIS MCD64A1 fire-history raster
│   ├── fetch_alfresco.py              # Fetch SNAP ALFRESCO flammability / veg-mode rasters
│   ├── build_prediction_data.py       # Build statewide 1 km datacube over roi.geojson -> prediction_data.nc
│   ├── roi.geojson                    # Main statewide region of interest
│   └── *.nc                           # Prediction datacubes & outputs (gitignored)
│
├── models/                            # Model training, prediction, interpretation
│   ├── spatial_cv.py                  # Buffered spatial-block CV splitter (Albers km-grid)
│   ├── train_xgboost.py               # Train XGBoost under nested spatial CV          -> model.json (+ config/manifest/hparams)
│   ├── predict.py                     # Score datacube -> statewide log-evidence susceptibility surface (Obu-masked)
│   ├── aoa.py                         # Area-of-Applicability reliability layer         -> aoa.nc
│   ├── shap_values.py                 # Pooled out-of-fold SHAP interpretation
│   ├── shap_groups.py                 # Grouped-family SHAP construction
│   ├── model.json                     # Operative all-data model (loaded by predict/aoa)
│   └── cv_config.json, run_manifest.json, selected_hparams.json, aoa_threshold.json   # Persisted run config
│
├── output/                            # Figure-generation scripts + rendered figure assets
│   ├── figstyle.py, STYLE.md          # Figure style guide (code-first; used by figure scripts)
│   ├── fig*_*.py                      # Manuscript figure generators
│   ├── archive/                       # Historical figures
│   └── *.png / *.pdf                  # Rendered figures, SHAP plots, prediction maps
│
├── diagnostics/                       # /verify-ml suite (baseline, leakage, parity, AOA probes) + FINDINGS.md
├── manuscript/                        # Earth's Future draft (main.tex, sections/, figures/) + STRATEGY/OUTLINE
│
├── archive/                           # Legacy code, data, and superseded models
│   ├── train_xgboost_previous_thawdb.py   # Legacy training on an older database
│   ├── keras-neural-network.py            # Archived alternative neural-network approach
│   └── data/, output/, TASKS.md           # Older data/scripts, SHAP outputs, closed task ledger (TNN decoder)
│
├── settings.py                        # Path config (ROOT/DATA/MODELS/OUTPUT/EE_PROJECT; imported by pipeline scripts)
├── CLAUDE.md, MAP.md                   # Project backbone + location index (start here)
├── SCOPE.md, PIPELINE.md, REFERENCES.md   # Manuscript scope, pipeline provenance, literature record
├── pyproject.toml                     # Poetry dependencies and project metadata
└── README.md                          # This file
```

## Usage

### Data Processing Pipeline

1. **Build feature table** (requires Google Earth Engine):
   ```bash
   poetry run python data/build_feature_table.py
   ```
   This extracts geospatial features (elevation, slope, land cover, climate variables, etc.) from Google Earth Engine for all points in the thaw database.

2. **Clean feature table**:
   ```bash
   poetry run python data/clean_feature_table.py
   ```
   This script removes unnecessary columns, handles missing values, encodes categorical variables, and prepares the data for machine learning.

### Model Training

Train the XGBoost model under nested spatial cross-validation:
```bash
poetry run python models/train_xgboost.py
```

This script:
- Runs nested spatial-block cross-validation (Albers km-grid blocks, `spatial_cv.py`) for honest out-of-sample evaluation
- Selects hyperparameters on pooled-OOF AUC-PR (positive = Non-abrupt); baseline diagnostics (dummy, logistic) run through the same folds
- Refits the operative all-data model and persists the CV config, run manifest, and selected hyperparameters
- Saves the trained model to `models/model.json`

### Model Interpretation

Generate SHAP values for model interpretability:
```bash
poetry run python models/shap_values.py
```

This refits the model per spatial fold and computes **pooled out-of-fold** TreeSHAP (attributions on held-out points), so the explanation is not read off the training data.

### Statewide Prediction

1. **Build the prediction datacube** (requires Google Earth Engine):
   ```bash
   poetry run python data/build_prediction_data.py
   ```
   This rasterizes all model features over the region of interest (`data/roi.geojson`, 1 km; terrain served natively) and writes `data/prediction_data.nc`.

2. **Generate the map**:
   ```bash
   poetry run python models/predict.py
   ```
   This scores the datacube with `models/model.json` and writes the statewide log-evidence susceptibility map (plus a diagnostic probability map) to `output/`, and the susceptibility/probability NetCDFs to `data/`, masked to the Obu permafrost domain. No discrete classification is produced (a single continuous surface is the sole map product).

3. **Compute the reliability layer**:
   ```bash
   poetry run python models/aoa.py
   ```
   This runs after `predict.py` on the same domain and writes the Area-of-Applicability layer (`data/aoa.nc`): a continuous dissimilarity index plus a derived extrapolation flag marking where the grid falls outside the training feature distribution.

> **Note:** `models/model.json` is the single operative model, refit on all data with the CV-selected hyperparameters. The old calibrated `.pkl` track is retired and not regenerated.

See `PIPELINE.md` for the full data-source → artifact provenance and execution order.

## Dependencies

Key dependencies include:
- `xgboost` - Gradient boosting framework
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `earthengine-api` - Google Earth Engine integration
- `shap` - Model interpretability
- `matplotlib`, `seaborn` - Visualization
- `keras`, `jax` - Deep learning (archived alternative approach; see `archive/`)
- `imbalanced-learn` - Handling class imbalance

See `pyproject.toml` for complete dependency list and versions.

## License

This project is licensed under the GPL-3.0-or-later License.

## Authors

- **Ethan Pierce** - ethan.g.pierce@dartmouth.edu

## Contributing

This is an active research project. For questions or contributions, please contact the project maintainers.
