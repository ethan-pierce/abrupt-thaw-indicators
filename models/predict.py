"""Generate predictions using the trained XGBoost model and prediction feature stack."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import xgboost as xgb

# Paths
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

# Load model
print(f"\nLoading model from: {model_path}")
model = xgb.XGBClassifier()
model.load_model(str(model_path))

# Extract feature names from model to verify order
with open(model_path, 'r') as f:
    model_json = json.load(f)
model_feature_names = model_json['learner']['feature_names']

print(f"Model loaded successfully")
print(f"Number of features expected by model: {len(model_feature_names)}")

# Load prediction data
print(f"\nLoading prediction data from: {prediction_data_path}")
ds = xr.open_dataset(prediction_data_path)

print(f"Dataset shape: {ds.dims}")
print(f"Feature stack shape: {ds['feature_stack'].shape}")

# Get feature stack and feature names from dataset
feature_stack = ds['feature_stack'].values  # Shape: (y, x, feature)
dataset_feature_names = ds['feature'].values.tolist()

print(f"Number of features in dataset: {len(dataset_feature_names)}")

# Verify feature names match
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
    
    # Reorder dataset features to match model order
    print("\nReordering dataset features to match model order...")
    feature_indices = [dataset_feature_names.index(name) for name in model_feature_names]
    feature_stack = feature_stack[:, :, feature_indices]
    print("Features reordered successfully")
else:
    print("Feature names match - proceeding with prediction")

# Get spatial dimensions
y_size, x_size, n_features = feature_stack.shape
n_pixels = y_size * x_size

print(f"\nSpatial dimensions: {y_size} x {x_size} = {n_pixels} pixels")

# Get default value from dataset attributes
default_value = ds.attrs.get('default_value', -9999)
print(f"Default (missing) value: {default_value}")

# Reshape feature stack for prediction: (y, x, feature) -> (n_pixels, feature)
print("\nReshaping feature stack for prediction...")
feature_array = feature_stack.reshape(n_pixels, n_features)

# Handle missing values (replace default_value with NaN, which XGBoost handles)
print("Handling missing values...")
feature_array = np.where(feature_array == default_value, np.nan, feature_array)

# --- Obu permafrost-domain mask [T20] ---------------------------------------------
# Replaces the old arbitrary ">= 50% of features non-NaN" keep. The target concept
# -- abrupt vs non-abrupt thaw -- is only DEFINED where permafrost exists, and the
# model (trained almost entirely on permafrost sites: 0.4% of training points fall
# below Obu PerProb 0.01) never saw non-permafrost negatives, so it cannot self-mask
# and would emit a confident, meaningless log-evidence off-domain. We therefore
# restrict the surface to the Obu permafrost domain, sampling Obu PerProb at each
# cell's persisted lon/lat (nearest -- the identical construction the datacube's
# LOCAL features use, so the mask lands on the same grid).
#
# Keep rule:  keep = (PerProb > 0) AND (>= 1 feature non-NaN)
#   * PerProb > 0 keeps the WHOLE permafrost domain incl. isolated permafrost: Obu
#     assigns exactly 0 to modeled non-permafrost and small positives to isolated,
#     so no epsilon is needed. The low threshold is deliberate -- PerProb is
#     label-entangled (non-abrupt lives in sporadic/discontinuous permafrost, median
#     PerProb ~0.37 vs ~0.94 for abrupt), so a higher cut would amputate the minority
#     class's home range. The mask is BINARY: PerProb never weights the surface
#     (weighting by it would systematically suppress the minority class).
#   * the >= 1-feature guard refuses to paint a base-rate pixel from all-NaN input
#     (XGBoost returns a finite base score, not NaN, on an all-missing row).
# Reliability / extrapolation is a SEPARATE concern (AOA, T21), not folded in here.
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

# Make predictions
print("\nGenerating predictions...")
print("  This may take a while for large datasets...")

# Predict probabilities (class 0 = abrupt thaw, class 1 = non-abrupt thaw)
# Use index 0 for abrupt thaw (majority class, ~94% of training data)
probabilities = model.predict_proba(feature_array)[:, 0]

# T19 [E13]: log-evidence susceptibility index -- the PRIMARY output surface.
#   log_evidence = logit(P_model(abrupt|x)) - logit(pi_sample(abrupt))
# A prior-free log-likelihood-ratio for abrupt vs non-abrupt thaw: 0 = neutral, >0 favours
# abrupt, <0 favours non-abrupt. This is NOT a calibrated probability and NOT a discrete
# class -- the sample prior is a lake-/road-biased sampling artifact and the landscape
# prior is unrecoverable, so only the prior-free evidence is defensible.
#
# pi_sample(abrupt) is the abrupt (class 0) fraction of the SAME features_clean.csv the
# operative model was refit on. With scale_pos_weight=1 (train_xgboost.py:96-98, T10)
# that sample prevalence is exactly the prior baked into P_model, so subtracting its
# logit divides the prior back out. Read at score time so it always tracks the refit
# data (currently ~0.9428; the older ~0.932 figure predates the v2 table).
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

# Additional validation: ensure predictions are finite (not NaN or inf)
finite_predictions = np.isfinite(probabilities)
valid_pixels = valid_pixels & finite_predictions
n_finite_invalid = (~finite_predictions).sum()
if n_finite_invalid > 0:
    print(f"Warning: {n_finite_invalid:,} pixels have non-finite predictions (NaN or inf) and will be excluded")

# Reshape predictions back to spatial dimensions
print("\nReshaping predictions to spatial dimensions...")
probabilities_2d = probabilities.reshape(y_size, x_size)
log_evidence_2d = log_evidence.reshape(y_size, x_size)

# Apply the domain mask to the SAVED products, not just the figures [T20]: outside
# the permafrost domain (or where a pixel has no data) susceptibility is undefined,
# so NaN it everywhere -- otherwise susceptibility.nc carries confident values over
# the ocean. NaN is the datacube's own missing convention, so "off-domain" is
# indistinguishable-by-design from "no data", which is the intended meaning.
invalid_mask = (~valid_pixels).reshape(y_size, x_size)
log_evidence_2d = np.where(invalid_mask, np.nan, log_evidence_2d)
probabilities_2d = np.where(invalid_mask, np.nan, probabilities_2d)

# Calculate prediction statistics (only for valid pixels with sufficient valid features and finite predictions)
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

# Create output dataset
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

# Save predictions to NetCDF
output_path = data_dir / 'predictions.nc'
print(f"\nSaving predictions to: {output_path}")
output_ds.to_netcdf(output_path)
print("Predictions saved successfully")

# Primary product: the log-evidence susceptibility surface on its own [T19/E13].
susceptibility_path = data_dir / 'susceptibility.nc'
susceptibility_ds = xr.Dataset(
    {'log_evidence': output_ds['log_evidence']}, coords=output_ds.coords, attrs=output_ds.attrs
)
susceptibility_ds.to_netcdf(susceptibility_path)
print(f"  Susceptibility (log-evidence) saved to: {susceptibility_path}")

# Also save the diagnostic probability surface as a separate file for easier access
prob_output_path = data_dir / 'prediction_probabilities.nc'
prob_ds = xr.Dataset({'probability': output_ds['probability']}, coords=output_ds.coords, attrs=output_ds.attrs)
prob_ds.to_netcdf(prob_output_path)
print(f"  Probabilities saved to: {prob_output_path}")

# Create map visualization
print("\nCreating probability map...")

# Geographic bounds from the datacube's own per-cell lon/lat coords [T20/T46].
# predict.py no longer reads roi.geojson (removed). Off-ROI cells carry a -9999 fill,
# so take the extent from finite, in-range coordinates only.
_ok = (np.isfinite(lon2d) & (np.abs(lon2d) <= 180)
       & np.isfinite(lat2d) & (np.abs(lat2d) <= 90))
lon_min, lon_max = float(lon2d[_ok].min()), float(lon2d[_ok].max())
lat_min, lat_max = float(lat2d[_ok].min()), float(lat2d[_ok].max())

print(f"Geographic bounds: Lon [{lon_min:.2f}, {lon_max:.2f}], Lat [{lat_min:.2f}, {lat_max:.2f}]")

output_dir = OUTPUT
output_dir.mkdir(exist_ok=True)

# Primary product map: log-evidence susceptibility, diverging colormap centred at 0 [T19/E13].
print("\nCreating log-evidence susceptibility map (primary product)...")
# log_evidence_2d is already masked at save time [T20]; np.where is a harmless no-op.
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

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# invalid_mask (the Obu domain + evidence mask) was computed once at save time [T20].
masked_prob = np.where(invalid_mask, np.nan, probabilities_2d)

# Plot probabilities
im = ax.imshow(
    np.flipud(masked_prob),  # Flip vertically to match geographic orientation
    extent=[lon_min, lon_max, lat_min, lat_max],
    cmap='RdYlBu_r',  # Red-Yellow-Blue reversed: red = high probability, blue = low
    aspect='auto',
    origin='lower',
    interpolation='nearest'
)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Probability of Abrupt Thaw', fraction=0.046, pad=0.04)
cbar.set_label('Probability of Abrupt Thaw', rotation=270, labelpad=20)

# Set labels and title
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Abrupt Thaw Probability', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.0, linestyle='--')

# Save map
map_output_path = output_dir / 'prediction_probability_map.png'
plt.savefig(map_output_path, dpi=600, bbox_inches='tight')
print(f"Probability map saved to: {map_output_path}")

plt.close('all')  # Close all figures to free memory

print("\n" + "="*80)
print("PREDICTION COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  - {susceptibility_path}  (PRIMARY: log-evidence susceptibility)")
print(f"  - {le_map_path}  (PRIMARY map)")
print(f"  - {output_path}")
print(f"  - {prob_output_path}")
print(f"  - {map_output_path}")
