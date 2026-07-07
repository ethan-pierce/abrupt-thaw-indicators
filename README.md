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

> **Note:** `models/model.json` is the operative model. A calibrated variant (`models/model_calibrated.pkl`) also exists; the choice between them, and the robustness of the statewide map, are open scientific questions tracked in `SCOPE.md` (they change ~37% of map pixels).

## To-Do List

### Higher Priority
- [ ] Compile cross-validation results across all model architectures
- [ ] Switch to configuration files for hyperparameters instead of hardcoding
- [ ] Formalize and document feature exclusion protocol 

### Lower Priority
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
