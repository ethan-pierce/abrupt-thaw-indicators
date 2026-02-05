"""Build a datacube of predictors over training lands."""

import ee
ee.Authenticate()
ee.Initialize(project = 'ee-abrupt-thaw')

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import geemap
import geopandas as gpd
import xarray as xr

data = Path(__file__).parent

with open(data / 'training-lands.geojson', 'r') as f:
    training_lands_json = json.load(f)
ee_training_lands_fc = geemap.geojson_to_ee(training_lands_json)

# Get geometry from FeatureCollection
if isinstance(ee_training_lands_fc, ee.FeatureCollection):
    # Get geometry from first feature
    first_feature = ee.Feature(ee_training_lands_fc.first())
    ee_training_lands = first_feature.geometry()
else:
    ee_training_lands = ee_training_lands_fc

SCALE = 500  # Use 1000m scale for testing

def load_data(image: ee.Image, projection, scale: float) -> ee.Image:
    """Load and rasterize a dataset."""
    return image.reproject(projection, scale=scale).clip(ee_training_lands)

def verify_grid_alignment(
    image1: ee.Image, 
    image2: ee.Image, 
    region: ee.Geometry,
    image1_default: float = None,
    image2_default: float = None
) -> bool:
    """Verify two images are on the exact same grid (CRS, transform, dimensions)."""
    p1, p2 = image1.projection().getInfo(), image2.projection().getInfo()
    crs_match = p1.get('crs') == p2.get('crs')
    transform_match = p1.get('transform') == p2.get('transform')
    
    # Unmask images if default values are provided
    img1 = image1.unmask(image1_default) if image1_default is not None else image1
    img2 = image2.unmask(image2_default) if image2_default is not None else image2
    
    s1 = img1.sampleRectangle(region=region)
    s2 = img2.sampleRectangle(region=region)
    b1, b2 = image1.bandNames().getInfo()[0], image2.bandNames().getInfo()[0]
    shape_match = np.array(s1.get(b1).getInfo()).shape == np.array(s2.get(b2).getInfo()).shape
    
    return crs_match and transform_match and shape_match

def extract_data_array(
    image: ee.Image, 
    region: ee.Geometry, 
    band_name: str = None,
    default_value: float = None
) -> np.ndarray:
    """Extract data array from a reprojected Earth Engine image (from load_data) as a numpy array."""
    # Unmask image with default value if provided
    img = image.unmask(default_value) if default_value is not None else image
    
    # Pass default value to sampleRectangle to handle pixels outside image footprint
    sampled = img.sampleRectangle(region=region, defaultValue=default_value)
    
    if band_name is None:
        band_name = image.bandNames().getInfo()[0]
    
    return np.array(sampled.get(band_name).getInfo(), dtype=float)

# Load model and extract feature names
model_path = Path(__file__).parent.parent / 'models' / 'model.json'
model = xgb.XGBClassifier()
model.load_model(str(model_path))

# Extract feature names from the model JSON
with open(model_path, 'r') as f:
    model_json = json.load(f)
feature_names = model_json['learner']['feature_names']

print(f"Model loaded from: {model_path}")
print(f"\nNumber of input features: {len(feature_names)}")
print(f"\nInput feature names:")
for i, name in enumerate(feature_names, 1):
    print(f"  {i:2d}. {name}")

def load_all_features(feature_names: list, scale: float, region: ee.Geometry, default_value: float = -9999) -> np.ndarray:
    """Load all features in the exact order required by the model and stack them for prediction."""
    feature_arrays = {}
    
    # Load elevation first to establish projection
    elevation_image = ee.Image('USGS/3DEP/10m').select('elevation')
    elevation = load_data(elevation_image, 'EPSG:4326', scale)
    projection = elevation.projection()
    
    if 'Elevation' in feature_names:
        feature_arrays['Elevation'] = extract_data_array(elevation, region, 'elevation', default_value)
    
    if 'Slope' in feature_names:
        slope = load_data(ee.Terrain.slope(elevation_image), projection, scale)
        slope_data = extract_data_array(slope, region, 'slope', default_value)
        # Flip vertically: Earth Engine returns data with first row = northernmost,
        # but we want first row = southernmost for consistency
        feature_arrays['Slope'] = np.flipud(slope_data)

    if 'Aspect' in feature_names:
        aspect = load_data(ee.Terrain.aspect(elevation_image), projection, scale)
        aspect_data = extract_data_array(aspect, region, 'aspect', default_value)
        feature_arrays['Aspect'] = np.flipud(aspect_data)
    
    # Load curvature features
    if 'Mean curvature (500 m)' in feature_names:
        curve500 = load_data(ee.Image('projects/ee-abrupt-thaw/assets/AK-curvature-500m').select('MeanCurvature'), projection, scale)
        curve500_data = extract_data_array(curve500, region, 'MeanCurvature', default_value)
        feature_arrays['Mean curvature (500 m)'] = np.flipud(curve500_data)

    if 'Mean curvature (2 km)' in feature_names:
        curve2k = load_data(ee.Image('projects/ee-abrupt-thaw/assets/AK-curvature-2k').select('MeanCurvature'), projection, scale)
        curve2k_data = extract_data_array(curve2k, region, 'MeanCurvature', default_value)
        feature_arrays['Mean curvature (2 km)'] = np.flipud(curve2k_data)
    
    # Load bioclimatic variables
    bioclim = ee.Image('WORLDCLIM/V1/BIO')
    bioclim_vars = {
        'Temperature Seasonality': 'bio04',
        'Temperature Annual Range': 'bio07',
        'Annual Precipitation': 'bio12',
        'Precipitation Seasonality': 'bio15'
    }
    for name, band in bioclim_vars.items():
        if name in feature_names:
            bioclim_img = load_data(bioclim.select(band), projection, scale)
            bioclim_data = extract_data_array(bioclim_img, region, band, default_value)
            feature_arrays[name] = np.flipud(bioclim_data)
    
    # Load other continuous features
    if 'Flammability Index' in feature_names:
        flammability = load_data(ee.Image('projects/ee-abrupt-thaw/assets/ALFRESCO-historical-flammability'), projection, scale)
        flammability_data = extract_data_array(flammability, region, 'b1', default_value)
        feature_arrays['Flammability Index'] = np.flipud(flammability_data)

    if 'Maximum Fire Temperature' in feature_names:
        firms = load_data(ee.Image('projects/ee-abrupt-thaw/assets/max-fire-temp'), projection, scale)
        firms_data = extract_data_array(firms, region, 'T21', default_value)
        feature_arrays['Maximum Fire Temperature'] = np.flipud(firms_data)

    if 'Mean Annual SWE' in feature_names:
        swe = load_data(ee.Image('projects/ee-abrupt-thaw/assets/ee-mean-annual-swe'), projection, scale)
        swe_data = extract_data_array(swe, region, 'swe', default_value)
        feature_arrays['Mean Annual SWE'] = np.flipud(swe_data)
    
    if 'Trend in SWE' in feature_names:
        swe_trend = load_data(ee.Image('projects/ee-abrupt-thaw/assets/annual-swe-trend').select('scale'), projection, scale)
        swe_trend_data = extract_data_array(swe_trend, region, 'scale', default_value)
        feature_arrays['Trend in SWE'] = np.flipud(swe_trend_data)

    if 'Projected precipitation change' in feature_names:
        precip_change = load_data(ee.Image('projects/ee-abrupt-thaw/assets/annual-precipitation-trend'), projection, scale)
        precip_change_data = extract_data_array(precip_change, region, 'b1', default_value)
        feature_arrays['Projected precipitation change'] = np.flipud(precip_change_data)
    
    # Load categorical features (Land Cover and Vegetation Mode) - one-hot encoded
    land_cover_labels = {
        11: 'Open Water',
        12: 'Perennial Ice/Snow',
        21: 'Developed, Open Space',
        22: 'Developed, Low Intensity',
        23: 'Developed, Medium Intensity',
        24: 'Developed, High Intensity',
        31: 'Barren Land (Rock/Sand/Clay)',
        41: 'Deciduous Forest',
        42: 'Evergreen Forest',
        43: 'Mixed Forest',
        51: 'Dwarf Scrub',
        52: 'Shrub/Scrub',
        71: 'Grassland/Herbaceous',
        72: 'Sedge/Herbaceous',
        81: 'Pasture/Hay',
        82: 'Cultivated Crops',
        90: 'Woody Wetlands',
        95: 'Emergent Herbaceous Wetlands'
    }
    
    if any('Land Cover' in name for name in feature_names):
        landcover = load_data(ee.Image('projects/ee-abrupt-thaw/assets/NLCD-2016'), projection, scale)
        landcover_array = extract_data_array(landcover, region, 'b1', default_value)
        
        for code, label in land_cover_labels.items():
            feature_name = f'Land Cover ({label})'
            if feature_name in feature_names:
                landcover_data = (landcover_array == code).astype(float)
                feature_arrays[feature_name] = np.flipud(landcover_data)

    vegetation_mode_labels = {
        1: 'Black spruce',
        2: 'White spruce',
        3: 'Deciduous forest',
        4: 'Shrub tundra',
        5: 'Graminoid tundra',
        6: 'Wetland tundra',
        7: 'Barren lichen moss',
        8: 'Temperate rainforest'
    }
    
    if any('Vegetation Mode' in name for name in feature_names):
        vegetation = load_data(ee.Image('projects/ee-abrupt-thaw/assets/ALFRESCO-historical-vegetation-mode'), projection, scale)
        vegetation_array = extract_data_array(vegetation, region, 'b1', default_value)
        
        for code, label in vegetation_mode_labels.items():
            feature_name = f'Vegetation Mode ({label})'
            if feature_name in feature_names:   
                vegetation_data = (vegetation_array == code).astype(float)
                feature_arrays[feature_name] = np.flipud(vegetation_data)
    
    # Load soil variables (need to aggregate depths)
    soil_vars = ['Soil Organic Carbon', 'Nitrogen', 'Bulk Density', 'Sand', 'Silt', 'Clay']
    soil_depths = {
        '0-5 cm': ('soc_0-5cm_mean', 'nitrogen_0-5cm_mean', 'bdod_0-5cm_mean', 'sand_0-5cm_mean', 'silt_0-5cm_mean', 'clay_0-5cm_mean'),
        '5-15 cm': ('soc_5-15cm_mean', 'nitrogen_5-15cm_mean', 'bdod_5-15cm_mean', 'sand_5-15cm_mean', 'silt_5-15cm_mean', 'clay_5-15cm_mean'),
        '15-30 cm': ('soc_15-30cm_mean', 'nitrogen_15-30cm_mean', 'bdod_15-30cm_mean', 'sand_15-30cm_mean', 'silt_15-30cm_mean', 'clay_15-30cm_mean'),
        '30-60 cm': ('soc_30-60cm_mean', 'nitrogen_30-60cm_mean', 'bdod_30-60cm_mean', 'sand_30-60cm_mean', 'silt_30-60cm_mean', 'clay_30-60cm_mean'),
        '60-100 cm': ('soc_60-100cm_mean', 'nitrogen_60-100cm_mean', 'bdod_60-100cm_mean', 'sand_60-100cm_mean', 'silt_60-100cm_mean', 'clay_60-100cm_mean'),
        '100-200 cm': ('soc_100-200cm_mean', 'nitrogen_100-200cm_mean', 'bdod_100-200cm_mean', 'sand_100-200cm_mean', 'silt_100-200cm_mean', 'clay_100-200cm_mean')
    }
    
    soil_images = {
        'Soil Organic Carbon': ee.Image('projects/soilgrids-isric/soc_mean'),
        'Nitrogen': ee.Image('projects/soilgrids-isric/nitrogen_mean'),
        'Bulk Density': ee.Image('projects/soilgrids-isric/bdod_mean'),
        'Sand': ee.Image('projects/soilgrids-isric/sand_mean'),
        'Silt': ee.Image('projects/soilgrids-isric/silt_mean'),
        'Clay': ee.Image('projects/soilgrids-isric/clay_mean')
    }
    
    # Load and aggregate soil variables
    for var in soil_vars:
        for depth_range in ['0-30 cm', '30-200 cm']:
            feature_name = f'{var} ({depth_range})'
            if feature_name in feature_names:
                if depth_range == '0-30 cm':
                    # Weighted average: (0-5)*5 + (5-15)*10 + (15-30)*15 / 30
                    depth_bands = ['0-5 cm', '5-15 cm', '15-30 cm']
                    weights = [5, 10, 15]
                else:  # 30-200 cm
                    # Weighted average: (30-60)*30 + (60-100)*40 + (100-200)*100 / 170
                    depth_bands = ['30-60 cm', '60-100 cm', '100-200 cm']
                    weights = [30, 40, 100]
                
                var_idx = soil_vars.index(var)
                arrays = []
                for depth, weight in zip(depth_bands, weights):
                    band = soil_depths[depth][var_idx]
                    img = load_data(soil_images[var].select(band), projection, scale)
                    arr = extract_data_array(img, region, band, default_value)
                    # Flip before aggregating
                    arr_flipped = np.flipud(arr)
                    arrays.append(arr_flipped * weight)
                
                total_weight = sum(weights)
                if depth_range == '0-30 cm':
                    feature_arrays[feature_name] = sum(arrays) / 30
                else:
                    feature_arrays[feature_name] = sum(arrays) / 170

    
    # Stack features in the exact order required by the model
    feature_stack = np.stack([feature_arrays[name] for name in feature_names], axis=-1)
    return feature_stack

# Get the bounding box of the training lands for extraction
training_lands_bounds = ee_training_lands.bounds()

# Load all features in model order and create feature stack
print("\nLoading all features for prediction over training lands...")
feature_stack = load_all_features(feature_names, SCALE, training_lands_bounds, default_value=-9999)

print(f"\nFeature stack shape: {feature_stack.shape}")
print(f"Expected shape: (height, width, {len(feature_names)})")

# Create xarray Dataset with feature stack and metadata
ds = xr.Dataset(
    {
        'feature_stack': (['y', 'x', 'feature'], feature_stack)
    },
    coords={
        'feature': feature_names,
        'x': np.arange(feature_stack.shape[1]),
        'y': np.arange(feature_stack.shape[0])
    },
    attrs={
        'scale': SCALE,
        'default_value': -9999,
        'description': 'Feature stack for abrupt thaw prediction model over training lands',
        'num_features': len(feature_names),
        'shape': f"{feature_stack.shape[0]} x {feature_stack.shape[1]} x {feature_stack.shape[2]}",
        'region': 'training-lands'
    }
)

# Add feature names as a coordinate variable for easy access
ds['feature_names'] = ('feature', feature_names)

# Save to NetCDF
feature_stack_path = data / 'prediction_data_traininglands.nc'
ds.to_netcdf(feature_stack_path)
print(f"\nFeature stack and metadata saved to: {feature_stack_path}")
print(f"  Shape: {feature_stack.shape}")
print(f"  Features: {len(feature_names)}")
print(f"  Scale: {SCALE}m")
print(f"  Region: Training lands")

