"""Generate predictions using the trained XGBoost model and prediction feature stack for training lands."""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xgboost as xgb

# Paths
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT

data_dir = DATA
models_dir = MODELS
model_path = models_dir / 'model.json'
prediction_data_path = data_dir / 'prediction_data_traininglands.nc'

print("="*80)
print("ABRUPT THAW PREDICTION (TRAINING LANDS)")
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

# Check for any pixels with all missing features
valid_pixels = ~np.isnan(feature_array).all(axis=1)
n_valid = valid_pixels.sum()
n_invalid = (~valid_pixels).sum()

print(f"Valid pixels: {n_valid:,} ({n_valid/n_pixels*100:.1f}%)")
print(f"Invalid pixels (all features missing): {n_invalid:,} ({n_invalid/n_pixels*100:.1f}%)")

# Make predictions
print("\nGenerating predictions...")
print("  This may take a while for large datasets...")

# Predict probabilities (class 0 = abrupt thaw, class 1 = gradual thaw)
probabilities = model.predict_proba(feature_array)[:, 0]  # Use index 0 for abrupt thaw (majority class)

# Predict binary classes
predictions = model.predict(feature_array)

print("Predictions completed")

# Reshape predictions back to spatial dimensions
print("\nReshaping predictions to spatial dimensions...")
probabilities_2d = probabilities.reshape(y_size, x_size)
predictions_2d = predictions.reshape(y_size, x_size)

# Calculate prediction statistics
print("\nPrediction Statistics:")
print(f"  Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
print(f"  Probability mean: {probabilities.mean():.4f}")
print(f"  Probability median: {np.median(probabilities):.4f}")
print(f"  Abrupt thaw predictions: {(predictions == 0).sum():,} ({(predictions == 0).sum()/n_pixels*100:.1f}%)")
print(f"  Gradual thaw predictions: {(predictions == 1).sum():,} ({(predictions == 1).sum()/n_pixels*100:.1f}%)")

# Create output dataset
print("\nCreating output dataset...")
output_ds = xr.Dataset(
    {
        'probability': (['y', 'x'], probabilities_2d),
        'prediction': (['y', 'x'], predictions_2d)
    },
    coords={
        'x': ds.coords['x'],
        'y': ds.coords['y']
    },
    attrs={
        'model_path': str(model_path),
        'prediction_data_path': str(prediction_data_path),
        'description': 'Abrupt thaw predictions from XGBoost model (training lands)',
        'probability_description': 'Probability of abrupt thaw (class 0)',
        'prediction_description': 'Binary prediction: 0=Abrupt Thaw, 1=Gradual Thaw',
        'default_value': default_value,
        'scale': ds.attrs.get('scale', 'unknown')
    }
)

# Save predictions to NetCDF
output_path = data_dir / 'predictions_traininglands.nc'
print(f"\nSaving predictions to: {output_path}")
output_ds.to_netcdf(output_path)
print("Predictions saved successfully")

# Also save as separate files for easier access
prob_output_path = data_dir / 'prediction_probabilities_traininglands.nc'
pred_output_path = data_dir / 'prediction_classes_traininglands.nc'

prob_ds = xr.Dataset({'probability': output_ds['probability']}, coords=output_ds.coords, attrs=output_ds.attrs)
pred_ds = xr.Dataset({'prediction': output_ds['prediction']}, coords=output_ds.coords, attrs=output_ds.attrs)

prob_ds.to_netcdf(prob_output_path)
pred_ds.to_netcdf(pred_output_path)

print(f"  Probabilities saved to: {prob_output_path}")
print(f"  Classes saved to: {pred_output_path}")

# Create map visualization
print("\nCreating probability map...")

# Load ROI to get geographic bounds
roi_path = data_dir / 'roi.geojson'
with open(roi_path, 'r') as f:
    roi_json = json.load(f)

# Extract bounds from ROI
# Handle both Polygon and MultiPolygon geometries
if roi_json['features'][0]['geometry']['type'] == 'Polygon':
    coords = roi_json['features'][0]['geometry']['coordinates'][0]
elif roi_json['features'][0]['geometry']['type'] == 'MultiPolygon':
    # Get all coordinates from all polygons
    coords = []
    for polygon in roi_json['features'][0]['geometry']['coordinates']:
        coords.extend(polygon[0])
else:
    coords = roi_json['features'][0]['geometry']['coordinates'][0]

lons = [c[0] for c in coords]
lats = [c[1] for c in coords]
lon_min, lon_max = min(lons), max(lons)
lat_min, lat_max = min(lats), max(lats)

print(f"Geographic bounds: Lon [{lon_min:.2f}, {lon_max:.2f}], Lat [{lat_min:.2f}, {lat_max:.2f}]")

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Mask invalid pixels (where all features were missing)
# Use the valid_pixels mask we created earlier
invalid_mask = (~valid_pixels).reshape(y_size, x_size)
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
ax.set_title('Abrupt Thaw Probability Map (Training Lands)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

# Save map
output_dir = OUTPUT
output_dir.mkdir(exist_ok=True)
map_output_path = output_dir / 'prediction_probability_map_traininglands.png'
plt.savefig(map_output_path, dpi=300, bbox_inches='tight')
print(f"Probability map saved to: {map_output_path}")

# Also create a map of binary predictions
fig2, ax2 = plt.subplots(figsize=(14, 10))

masked_pred = np.where(invalid_mask, np.nan, predictions_2d)

im2 = ax2.imshow(
    np.flipud(masked_pred),
    extent=[lon_min, lon_max, lat_min, lat_max],
    cmap='RdYlGn',  # Red-Yellow-Green: red = abrupt (0), green = gradual (1)
    aspect='auto',
    origin='lower',
    interpolation='nearest',
    vmin=0,
    vmax=1
)

cbar2 = plt.colorbar(im2, ax=ax2, label='Thaw Type', fraction=0.046, pad=0.04, ticks=[0, 1])
cbar2.set_ticklabels(['Abrupt', 'Gradual'])  # 0=Abrupt, 1=Gradual
cbar2.set_label('Thaw Type', rotation=270, labelpad=20)

ax2.set_xlabel('Longitude (°E)', fontsize=12)
ax2.set_ylabel('Latitude (°N)', fontsize=12)
ax2.set_title('Abrupt Thaw Classification Map (Training Lands)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

map_output_path2 = output_dir / 'prediction_classification_map_traininglands.png'
plt.savefig(map_output_path2, dpi=300, bbox_inches='tight')
print(f"Classification map saved to: {map_output_path2}")

plt.close('all')  # Close all figures to free memory

print("\n" + "="*80)
print("PREDICTION COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  - {output_path}")
print(f"  - {prob_output_path}")
print(f"  - {pred_output_path}")
print(f"  - {map_output_path}")
print(f"  - {map_output_path2}")

