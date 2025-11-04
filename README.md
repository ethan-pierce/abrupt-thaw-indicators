# Using Machine Learning Models to Identify Indicators of Abrupt Thaw

A machine learning (ML) project focused on identifying indicators of abrupt permafrost thaw using geospatial features extracted from remote sensing, community data products, and climate model reanalysis.

## Overview

This repository contains tools and models for:
- Post-processing the Thaw Database to prepare it for ML applications 
- Compiling and extracting geospatial features from Google Earth Engine
- Training and evaluating multiple different ML classification models
- Analyzing model interpretability using Shapley (SHAP) values

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

- Python ≥3.13
- Poetry (install from [python-poetry.org](https://python-poetry.org/docs/#installation))

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
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
├── data/                          # Data processing and feature extraction
│   ├── Alaska_Permafrost_Thaw_Database_v1.0.0-alpha.csv  # Alpha version of the ThawDatabase
│   ├── Database_final_v1.csv      # Current version of the ThawDatabase
│   ├── features.csv               # Raw feature table
│   ├── features_clean.csv         # Cleaned feature table (used for training)
│   ├── build_feature_table.py     # Script used to extract features from Google Earth Engine
│   └── clean_feature_table.py     # Script used to clean and preprocess feature table
│
├── models/                       # Machine learning models and training
│   ├── train_xgboost.py          # Train XGBoost models and perform cross-validation tests
│   ├── shap_values.py            # Generate SHAP values for model interpretation
│   ├── model.json                # Trained XGBoost model (serialized)
│   └── model_previous_best.json  # Previous best model version
│
├── output/                       # Generated outputs and visualizations
│   ├── archive/                  # Historical outputs
│   │   ├── confusion-matrix.png
│   │   ├── feature-importance-*.png
│   │   ├── precision-recall.png
│   │   └── shap-values.png
│   └── MURI-update-July2025.pptx # Project update slides (summer 2025)
│
├── archive/                      # Legacy code and experimental work
│   ├── data/                     # Older data processing scripts
│   ├── keras-neural-network.py   # Alternative feedforward neural network approach
│   └── n500/                     # SHAP analysis outputs
│
├── settings.py                    # Path configuration
├── pyproject.toml                 # Poetry dependencies and project metadata
└── README.md                      # This file
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

## Model Details

### Current Best Model

- **Algorithm**: XGBoost Classifier
- **F1 Score**: 0.8818
- **Hyperparameters**:
  - `n_estimators`: 125
  - `max_depth`: 9
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `min_child_weight`: 5
  - `scale_pos_weight`: 30
  - `base_score`: 0.94

## To-Do List

### Higher Priority
- [ ] Interpolate SHAP results back to statewide feature maps
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
