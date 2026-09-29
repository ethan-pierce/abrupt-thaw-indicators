"""Generate predictions using the trained XGBoost model and prediction feature stack."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT
from data import local_rasters

data_dir = DATA
models_dir = MODELS
model_path = models_dir / 'model.json'
prediction_data_path = data_dir / 'prediction_data.nc'

print("="*80)
print("ABRUPT THAW PREDICTION")
print("="*80)

print(f"\nLoading model from: {model_path}")
model = xgb.XGBClassifier()
model.load_model(str(model_path))

with open(model_path, 'r') as f:
    model_json = json.load(f)
model_feature_names = model_json['learner']['feature_names']

print(f"Model loaded successfully")
print(f"Number of features expected by model: {len(model_feature_names)}")

print(f"\nLoading prediction data from: {prediction_data_path}")
ds = xr.open_dataset(prediction_data_path)

print(f"Dataset shape: {ds.dims}")
print(f"Feature stack shape: {ds['feature_stack'].shape}")

feature_stack = ds['feature_stack'].values  # Shape: (y, x, feature)
dataset_feature_names = ds['feature'].values.tolist()

print(f"Number of features in dataset: {len(dataset_feature_names)}")

if model_feature_names != dataset_feature_names:
    print("\nWARNING: Feature names don't match exactly!")
    print("Model features:")
    for i, name in enumerate(model_feature_names[:5]):
        print(f"  {i}: {name}")
    print(f"  ... ({len(model_feature_names)} total)")
    print("Dataset features:")
    for i, name in enumerate(dataset_feature_names[:5]):
        print(f"  {i}: {name}")
    print(f"  ... ({len(dataset_feature_names)} total)")
    
    print("\nReordering dataset features to match model order...")
    feature_indices = [dataset_feature_names.index(name) for name in model_feature_names]
    feature_stack = feature_stack[:, :, feature_indices]
    print("Features reordered successfully")
else:
    print("Feature names match - proceeding with prediction")

y_size, x_size, n_features = feature_stack.shape
n_pixels = y_size * x_size

print(f"\nSpatial dimensions: {y_size} x {x_size} = {n_pixels} pixels")

default_value = ds.attrs.get('default_value', -9999)
print(f"Default (missing) value: {default_value}")

print("\nReshaping feature stack for prediction...")
feature_array = feature_stack.reshape(n_pixels, n_features)

print("Handling missing values...")
feature_array = np.where(feature_array == default_value, np.nan, feature_array)

# --- Obu permafrost-domain mask ----------------------------------------------------
# Abrupt vs non-abrupt thaw is only defined where permafrost exists, and the model
# (0.4% of training points fall below Obu PerProb 0.01) never saw non-permafrost sites,
# so it would emit confident, meaningless log-evidence off-domain. Obu PerProb is
# sampled at each cell's lon/lat (nearest, as for the datacube's local features).
#
# keep = (PerProb > 0) AND (>= 1 feature non-NaN)
#   * Obu assigns exactly 0 to non-permafrost and small positives to isolated
#     permafrost, so no epsilon is needed. The low cut is deliberate: PerProb is
#     label-entangled (median ~0.37 for non-abrupt vs ~0.94 for abrupt), so a higher
#     cut would remove the minority class's range. The mask is binary; PerProb never
#     weights the surface.
#   * XGBoost returns a finite base score on an all-missing row, so all-NaN pixels
#     are dropped.
# Extrapolation is handled separately by the AOA layer (models/aoa.py).
if 'longitude' not in ds.coords or 'latitude' not in ds.coords:
    raise SystemExit(
        "prediction_data.nc has no longitude/latitude coords -- the Obu mask needs "
        "per-cell coordinates. Rebuild the datacube with the current "
        "data/build_prediction_data.py (T20/T46) before running predict.py."
    )
lon2d = ds['longitude'].values
lat2d = ds['latitude'].values
perprob = local_rasters.sample_points(
    local_rasters.OBU_TIF, lon2d.ravel(), lat2d.ravel()
).reshape(y_size, x_size)
in_domain = (perprob > 0)  # NaN (ocean / off Obu-coverage) and 0 (non-permafrost) -> False

n_valid_features_per_pixel = (~np.isnan(feature_array)).sum(axis=1)
has_evidence_2d = (n_valid_features_per_pixel >= 1).reshape(y_size, x_size)
valid_pixels = (in_domain & has_evidence_2d).reshape(n_pixels)
n_valid = valid_pixels.sum()
n_invalid = (~valid_pixels).sum()

print(f"Obu permafrost domain (PerProb > 0): {int(in_domain.sum()):,} pixels "
      f"({in_domain.sum()/n_pixels*100:.1f}%)")
print(f"Valid pixels (in-domain AND >=1 feature): {n_valid:,} ({n_valid/n_pixels*100:.1f}%)")
print(f"Masked pixels (off-domain or no data): {n_invalid:,} ({n_invalid/n_pixels*100:.1f}%)")

print("\nGenerating predictions...")
print("  This may take a while for large datasets...")

# class 0 = Abrupt (majority), class 1 = Non-abrupt
probabilities = model.predict_proba(feature_array)[:, 0]

# Log-evidence susceptibility, the primary output surface:
#   log_evidence = logit(P_model(abrupt|x)) - logit(pi_sample(abrupt))
# 0 = neutral, >0 favours abrupt. Not a calibrated probability: the sample prior is a
# lake-/road-biased sampling artifact and the landscape prior is unrecoverable.
#
# pi_sample is the abrupt fraction of the features_clean.csv the model was fit on. With
# scale_pos_weight=1 that is the prior baked into P_model, so subtracting its logit
# removes it.
pi_sample = float(
    (pd.read_csv(data_dir / 'features_clean.csv', usecols=['Class'])['Class'] == 0).mean()
)

def _logit(p, eps=1e-7):
    """Numerically safe logit; clips to (eps, 1-eps) so p in {0,1} stays finite."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

log_evidence = _logit(probabilities) - _logit(pi_sample)
print(f"Sample prior pi_sample(abrupt) = {pi_sample:.4f} (logit = {_logit(pi_sample):.4f})")

print("Predictions completed")

# Exclude non-finite predictions
finite_predictions = np.isfinite(probabilities)
valid_pixels = valid_pixels & finite_predictions
n_finite_invalid = (~finite_predictions).sum()
if n_finite_invalid > 0:
    print(f"Warning: {n_finite_invalid:,} pixels have non-finite predictions (NaN or inf) and will be excluded")

print("\nReshaping predictions to spatial dimensions...")
probabilities_2d = probabilities.reshape(y_size, x_size)
log_evidence_2d = log_evidence.reshape(y_size, x_size)

# Mask the saved products, not just the figures. Off-domain and no-data share the
# datacube's NaN missing convention by design.
invalid_mask = (~valid_pixels).reshape(y_size, x_size)
log_evidence_2d = np.where(invalid_mask, np.nan, log_evidence_2d)
probabilities_2d = np.where(invalid_mask, np.nan, probabilities_2d)

# Prediction statistics over valid pixels
print("\nPrediction Statistics (excluding invalid data):")
valid_probabilities = probabilities[valid_pixels]
n_valid_predictions = len(valid_probabilities)

if n_valid_predictions > 0:
    print(f"  Valid predictions: {n_valid_predictions:,} pixels")
    print(f"  Probability range: [{valid_probabilities.min():.4f}, {valid_probabilities.max():.4f}]")
    print(f"  Probability mean: {valid_probabilities.mean():.4f}")
    print(f"  Probability median: {np.median(valid_probabilities):.4f}")
    valid_log_evidence = log_evidence[valid_pixels]
    print(f"  Log-evidence range: [{valid_log_evidence.min():.3f}, {valid_log_evidence.max():.3f}] (0 = neutral)")
    print(f"  Log-evidence median: {np.median(valid_log_evidence):.3f}")
    print(f"  Pixels favouring abrupt (log-evidence > 0): {(valid_log_evidence > 0).sum():,} "
          f"({(valid_log_evidence > 0).sum()/n_valid_predictions*100:.1f}%)")
else:
    print("  WARNING: No valid predictions found!")

print("\nCreating output dataset...")
output_ds = xr.Dataset(
    {
        'log_evidence': (['y', 'x'], log_evidence_2d),
        'probability': (['y', 'x'], probabilities_2d)
    },
    coords={
        'x': ds.coords['x'],
        'y': ds.coords['y'],
        'longitude': ds.coords['longitude'],
        'latitude': ds.coords['latitude'],
    },
    attrs={
        'model_path': str(model_path),
        'prediction_data_path': str(prediction_data_path),
        'description': 'Abrupt-thaw susceptibility (log-evidence) from XGBoost model',
        'log_evidence_description': ('Primary surface [E13]: logit(P_model(abrupt|x)) '
                                     '- logit(pi_sample(abrupt)); 0 = neutral, >0 favours '
                                     'abrupt. Prior-free log-likelihood-ratio index, NOT a '
                                     'calibrated probability and NOT a discrete class.'),
        'pi_sample_abrupt': pi_sample,
        'probability_description': 'Diagnostic only: P_model(abrupt, class 0), calibrated to the sample prior',
        'domain_mask_description': ('[T20] Off-permafrost pixels are NaN: kept iff Obu PerProb '
                                    '(UiO_PEX_PERPROB_5.0) > 0 at the cell centre AND >=1 feature '
                                    'is non-NaN. Concept-validity mask (permafrost domain), '
                                    'binary -- PerProb does NOT weight the surface.'),
        'default_value': default_value,
        'scale': ds.attrs.get('scale', 'unknown')
    }
)

output_path = data_dir / 'predictions.nc'
print(f"\nSaving predictions to: {output_path}")
output_ds.to_netcdf(output_path)
print("Predictions saved successfully")

# Primary product: the log-evidence susceptibility surface on its own.
susceptibility_path = data_dir / 'susceptibility.nc'
susceptibility_ds = xr.Dataset(
    {'log_evidence': output_ds['log_evidence']}, coords=output_ds.coords, attrs=output_ds.attrs
)
susceptibility_ds.to_netcdf(susceptibility_path)
print(f"  Susceptibility (log-evidence) saved to: {susceptibility_path}")

# Diagnostic probability surface as a separate file
prob_output_path = data_dir / 'prediction_probabilities.nc'
prob_ds = xr.Dataset({'probability': output_ds['probability']}, coords=output_ds.coords, attrs=output_ds.attrs)
prob_ds.to_netcdf(prob_output_path)
print(f"  Probabilities saved to: {prob_output_path}")

print("\nCreating probability map...")

# Off-ROI cells carry a -9999 lon/lat fill, so take the extent from finite, in-range
# coordinates only.
_ok = (np.isfinite(lon2d) & (np.abs(lon2d) <= 180)
       & np.isfinite(lat2d) & (np.abs(lat2d) <= 90))
lon_min, lon_max = float(lon2d[_ok].min()), float(lon2d[_ok].max())
lat_min, lat_max = float(lat2d[_ok].min()), float(lat2d[_ok].max())

print(f"Geographic bounds: Lon [{lon_min:.2f}, {lon_max:.2f}], Lat [{lat_min:.2f}, {lat_max:.2f}]")

output_dir = OUTPUT
output_dir.mkdir(exist_ok=True)

# Primary product map: log-evidence susceptibility, diverging colormap centred at 0.
print("\nCreating log-evidence susceptibility map (primary product)...")
# log_evidence_2d is already masked; np.where is a no-op.
masked_log_evidence = np.where(invalid_mask, np.nan, log_evidence_2d)
le_absmax = float(np.nanmax(np.abs(masked_log_evidence))) if np.isfinite(masked_log_evidence).any() else 1.0

fig0, ax0 = plt.subplots(figsize=(14, 10))
im0 = ax0.imshow(
    np.flipud(masked_log_evidence),
    extent=[lon_min, lon_max, lat_min, lat_max],
    cmap='RdBu_r',  # red = positive (favours abrupt), white = 0 (neutral), blue = favours non-abrupt
    aspect='auto',
    origin='lower',
    interpolation='nearest',
    vmin=-le_absmax,
    vmax=le_absmax,
)
cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
cbar0.set_label('Abrupt-thaw log-evidence (0 = neutral, >0 favours abrupt)', rotation=270, labelpad=20)
ax0.set_xlabel('Longitude (°E)', fontsize=12)
ax0.set_ylabel('Latitude (°N)', fontsize=12)
ax0.set_title('Abrupt-Thaw Susceptibility (log-evidence)', fontsize=14, fontweight='bold')
le_map_path = output_dir / 'susceptibility_log_evidence_map.png'
plt.savefig(le_map_path, dpi=600, bbox_inches='tight')
print(f"Log-evidence susceptibility map saved to: {le_map_path}")

fig, ax = plt.subplots(figsize=(14, 10))

masked_prob = np.where(invalid_mask, np.nan, probabilities_2d)

im = ax.imshow(
    np.flipud(masked_prob),
    extent=[lon_min, lon_max, lat_min, lat_max],
    cmap='RdYlBu_r',
    aspect='auto',
    origin='lower',
    interpolation='nearest'
)

cbar = plt.colorbar(im, ax=ax, label='Probability of Abrupt Thaw', fraction=0.046, pad=0.04)
cbar.set_label('Probability of Abrupt Thaw', rotation=270, labelpad=20)

ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Abrupt Thaw Probability', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.0, linestyle='--')

map_output_path = output_dir / 'prediction_probability_map.png'
plt.savefig(map_output_path, dpi=600, bbox_inches='tight')
print(f"Probability map saved to: {map_output_path}")

plt.close('all')

print("\n" + "="*80)
print("PREDICTION COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  - {susceptibility_path}  (PRIMARY: log-evidence susceptibility)")
print(f"  - {le_map_path}  (PRIMARY map)")
print(f"  - {output_path}")
print(f"  - {prob_output_path}")
print(f"  - {map_output_path}")
