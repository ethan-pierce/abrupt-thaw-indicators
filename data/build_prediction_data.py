"""Build a datacube of predictors over interior and Arctic Alaska."""

import ee
ee.Authenticate()
ee.Initialize(project = 'ee-abrupt-thaw')

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import geemap

data = Path(__file__).parent

with open(data / 'roi.geojson', 'r') as f:
    roi_json = json.load(f)
ee_roi = geemap.geojson_to_ee(roi_json)

SCALE = 4000

# Load and rasterize NLCD data
nlcd = ee.Image('projects/ee-abrupt-thaw/assets/NLCD-2016')
nlcd_rasterized = nlcd.reproject('EPSG:4326', scale=SCALE).clip(ee_roi)

# Get the projection from the rasterized NLCD to ensure exact grid matching
nlcd_projection = nlcd_rasterized.projection()

# Load curvature 2k dataset and rasterize to the exact same grid as NLCD
curvature = ee.Image('projects/ee-abrupt-thaw/assets/AK-curvature-2k').select('MeanCurvature')
curvature_rasterized = curvature.reproject(nlcd_projection).clip(ee_roi)

# Verify both rasters are on the exact same grid
print("Verifying grid alignment...")
print("\n=== Projection Information ===")
nlcd_proj_info = nlcd_rasterized.projection().getInfo()
curvature_proj_info = curvature_rasterized.projection().getInfo()

print(f"NLCD CRS: {nlcd_proj_info.get('crs')}")
print(f"Curvature CRS: {curvature_proj_info.get('crs')}")
print(f"CRS Match: {nlcd_proj_info.get('crs') == curvature_proj_info.get('crs')}")

print(f"\nNLCD Transform: {nlcd_proj_info.get('transform')}")
print(f"Curvature Transform: {curvature_proj_info.get('transform')}")
print(f"Transform Match: {nlcd_proj_info.get('transform') == curvature_proj_info.get('transform')}")

# Calculate and display nominal resolution
transform = nlcd_proj_info.get('transform')
scale_x_deg = transform[0]  # pixel width in degrees
scale_y_deg = abs(transform[4])  # pixel height in degrees

# Get approximate center latitude for meter conversion
roi_bounds = ee_roi.bounds().getInfo()['coordinates'][0]
center_lat = np.mean([coord[1] for coord in roi_bounds])

# Convert degrees to meters (approximate, using WGS84)
# 1 degree latitude ≈ 111,320 meters (constant)
# 1 degree longitude ≈ 111,320 * cos(latitude) meters
meters_per_deg_lat = 111320
meters_per_deg_lon = 111320 * np.cos(np.radians(center_lat))

scale_x_m = scale_x_deg * meters_per_deg_lon
scale_y_m = scale_y_deg * meters_per_deg_lat

print(f"\n=== Raster Resolution ===")
print(f"Nominal scale (requested): {SCALE} meters")
print(f"Pixel size (X): {scale_x_deg:.6f} degrees ≈ {scale_x_m:.1f} meters")
print(f"Pixel size (Y): {scale_y_deg:.6f} degrees ≈ {scale_y_m:.1f} meters")
print(f"Average pixel size: {(scale_x_m + scale_y_m) / 2:.1f} meters")

# Sample both images to get actual dimensions
print("\n=== Raster Dimensions ===")
# Use defaultValue in sampleRectangle to handle fully masked pixels
# Use 0 as default for NLCD (land cover codes start from 11, so 0 is safe for masked areas)
sampled_nlcd = nlcd_rasterized.sampleRectangle(region=ee_roi, defaultValue=0)

# Use -9999 as default for curvature (as used in plot_input_data.py)
sampled_curvature = curvature_rasterized.sampleRectangle(region=ee_roi, defaultValue=-9999)

data_nlcd = np.array(sampled_nlcd.get('b1').getInfo(), dtype=float)
data_curvature = np.array(sampled_curvature.get('MeanCurvature').getInfo(), dtype=float)

print(f"NLCD shape: {data_nlcd.shape}")
print(f"Curvature shape: {data_curvature.shape}")
print(f"Shapes Match: {data_nlcd.shape == data_curvature.shape}")

# Generate coordinate arrays from projection and data dimensions
# Get the bounds of the ROI
roi_bounds = ee_roi.bounds().getInfo()['coordinates'][0]
min_lon = min(coord[0] for coord in roi_bounds)
max_lon = max(coord[0] for coord in roi_bounds)
min_lat = min(coord[1] for coord in roi_bounds)
max_lat = max(coord[1] for coord in roi_bounds)

# Get the transform to calculate pixel size
transform = nlcd_proj_info.get('transform')
scale_x = transform[0]  # pixel width in degrees
scale_y = abs(transform[4])  # pixel height in degrees (usually negative)

# Generate coordinate arrays based on data shape
rows, cols = data_nlcd.shape
lon_array = np.linspace(min_lon + scale_x/2, max_lon - scale_x/2, cols)
lat_array = np.linspace(max_lat - scale_y/2, min_lat + scale_y/2, rows)

# Create 2D coordinate grids
lon_nlcd = np.tile(lon_array, (rows, 1))
lat_nlcd = np.tile(lat_array.reshape(-1, 1), (1, cols))

# Use the same coordinates for both datasets since they're on the same grid
lon_curvature = lon_nlcd.copy()
lat_curvature = lat_nlcd.copy()

print(f"\n=== Coordinate Arrays ===")
print(f"NLCD lon shape: {lon_nlcd.shape}, lat shape: {lat_nlcd.shape}")
print(f"Curvature lon shape: {lon_curvature.shape}, lat shape: {lat_curvature.shape}")
print(f"Lon arrays match: {np.allclose(lon_nlcd, lon_curvature)}")
print(f"Lat arrays match: {np.allclose(lat_nlcd, lat_curvature)}")

print("\n=== Summary ===")
all_match = (
    nlcd_proj_info.get('crs') == curvature_proj_info.get('crs') and
    nlcd_proj_info.get('transform') == curvature_proj_info.get('transform') and
    data_nlcd.shape == data_curvature.shape
)
print(f"Grids are aligned: {all_match}")

# Visualize NLCD data on a map
print("\n=== Creating NLCD Map ===")
# Mask out 0 values (masked pixels) for visualization
data_nlcd_vis = data_nlcd.copy()
data_nlcd_vis[data_nlcd_vis == 0] = np.nan

fig, ax = plt.subplots(figsize=(14, 10))
# Use a qualitative colormap suitable for categorical land cover data
im = ax.imshow(
    np.flipud(data_nlcd_vis), 
    extent=[min_lon, max_lon, min_lat, max_lat], 
    cmap='tab20', 
    aspect='auto', 
    origin='lower',
    interpolation='nearest'
)
plt.colorbar(im, ax=ax, label='NLCD Land Cover Code')
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('NLCD 2016 Land Cover')
ax.grid(True, alpha=0.3)

# Save the map
output_path = data.parent / 'output' / 'nlcd_map.png'
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Map saved to: {output_path}")
plt.show()

# Visualize curvature data on a map
print("\n=== Creating Curvature Map ===")
# Mask out -9999 values (masked pixels) for visualization
data_curvature_vis = data_curvature.copy()
data_curvature_vis[data_curvature_vis == -9999] = np.nan

fig, ax = plt.subplots(figsize=(14, 10))
# Use a diverging colormap suitable for curvature data (as in plot_input_data.py)
im = ax.imshow(
    np.flipud(data_curvature_vis), 
    extent=[min_lon, max_lon, min_lat, max_lat], 
    cmap='RdYlBu_r', 
    aspect='auto', 
    origin='lower',
    interpolation='nearest',
    vmin = -1e-6,
    vmax = 1e-6
)
plt.colorbar(im, ax=ax, label='Mean Curvature (2 km)')
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('Mean Curvature (2 km)')
ax.grid(True, alpha=0.3)

# Save the map
output_path = data.parent / 'output' / 'curvature_map.png'
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Map saved to: {output_path}")
plt.show()


