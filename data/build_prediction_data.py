"""Build a datacube of predictors over interior and Arctic Alaska.

Same two-track sourcing as build_feature_table.py (no custom GEE assets, see
TASKS T0): public-catalog + re-derived GEE layers come from ``gee_features.py``
(curvature, SWE + trends, max fire temperature), and the four features with no
GEE-catalog upstream (ALFRESCO flammability + vegetation mode, NLCD land cover,
SNAP projected change) are nearest-sampled from local rasters at the datacube's
own cell centres via ``local_rasters.py``. No ``ASSET_ROOT`` dependency.
"""

import ee

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import EE_PROJECT

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import geemap
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA, MODELS
import gee_features
import local_rasters

data = DATA

with open(data / 'roi.geojson', 'r') as f:
    roi_json = json.load(f)
ee_roi = geemap.geojson_to_ee(roi_json)

SCALE = 4000

def load_data(image: ee.Image, projection, scale: float) -> ee.Image:
    """Load and rasterize a dataset."""
    return image.reproject(projection, scale=scale).clip(ee_roi)

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
model_path = MODELS / 'model.json'
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

    # LOCAL track: the per-cell WGS84 lon/lat of the datacube grid, so local
    # rasters can be nearest-sampled at the exact same cell centres as the GEE
    # layers. Extracted in GEE-native orientation (no flip); local arrays are
    # sampled at these coords and then np.flipud'd to match the flipped GEE
    # features below.
    lonlat = load_data(ee.Image.pixelLonLat(), projection, scale)
    lon2d = extract_data_array(lonlat, region, 'longitude', default_value)
    lat2d = extract_data_array(lonlat, region, 'latitude', default_value)

    def sample_local(path):
        """Nearest-sample a local raster onto the datacube grid (native
        orientation; caller flips to match the GEE features)."""
        flat = local_rasters.sample_points(path, lon2d.ravel(), lat2d.ravel())
        return flat.reshape(lon2d.shape)

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
    
    # Load curvature features (GEE track: TAGEE-family port, no custom asset)
    if 'Mean curvature (500 m)' in feature_names:
        curve500 = load_data(gee_features.mean_curvature(500).select('MeanCurvature'), projection, scale)
        curve500_data = extract_data_array(curve500, region, 'MeanCurvature', default_value)
        feature_arrays['Mean curvature (500 m)'] = np.flipud(curve500_data)

    if 'Mean curvature (2 km)' in feature_names:
        curve2k = load_data(gee_features.mean_curvature(2000).select('MeanCurvature'), projection, scale)
        curve2k_data = extract_data_array(curve2k, region, 'MeanCurvature', default_value)
        feature_arrays['Mean curvature (2 km)'] = np.flipud(curve2k_data)
    
    # Load bioclimatic variables
    bioclim = ee.Image('WORLDCLIM/V1/BIO')
    bioclim_vars = {
        'Annual Mean Temperature': 'bio01',
        'Mean Diurnal Range': 'bio02',
        'Isothermality': 'bio03',
        'Temperature Seasonality': 'bio04',
        'Max Temperature of Warmest Month': 'bio05',
        'Min Temperature of Coldest Month': 'bio06',
        'Temperature Annual Range': 'bio07',
        'Mean Temperature of Wettest Quarter': 'bio08',
        'Mean Temperature of Driest Quarter': 'bio09',
        'Mean Temperature of Warmest Quarter': 'bio10',
        'Mean Temperature of Coldest Quarter': 'bio11',
        'Annual Precipitation': 'bio12',
        'Precipitation of Wettest Month': 'bio13',
        'Precipitation of Driest Month': 'bio14',
        'Precipitation Seasonality': 'bio15',
        'Precipitation of Wettest Quarter': 'bio16',
        'Precipitation of Driest Quarter': 'bio17',
        'Precipitation of Warmest Quarter': 'bio18',
        'Precipitation of Coldest Quarter': 'bio19'
    }
    for name, band in bioclim_vars.items():
        if name in feature_names:
            bioclim_img = load_data(bioclim.select(band), projection, scale)
            bioclim_data = extract_data_array(bioclim_img, region, band, default_value)
            feature_arrays[name] = np.flipud(bioclim_data)
    
    # Flammability Index (LOCAL track): nearest-sample ALFRESCO at cell centres.
    if 'Flammability Index' in feature_names:
        feature_arrays['Flammability Index'] = np.flipud(sample_local(local_rasters.FLAMMABILITY_TIF))

    # Maximum fire temperature + Fire Detected (GEE track: FIRMS, no asset)
    if 'Maximum Fire Temperature' in feature_names:
        firms = load_data(gee_features.max_fire_temp(), projection, scale)
        firms_data = extract_data_array(firms, region, 'T21', default_value)
        feature_arrays['Maximum Fire Temperature'] = np.flipud(firms_data)

    if 'Fire Detected' in feature_names:
        # Train/serve parity with clean_feature_table.py (A1): Fire Detected = 1 where
        # FIRMS reported a maximum fire temperature (T21 present), 0 where the pixel is
        # masked (genuine "no fire") — the datacube analogue of
        # `notna(Maximum Fire Temperature)`. Reproject first, then take the mask so the
        # indicator is evaluated on the model's 4 km grid, matching the footprint that
        # the Maximum Fire Temperature layer treats as valid-vs-missing. Use 0 as the
        # fill so unobserved pixels read as "no fire", never -9999.
        firms_binary = load_data(
            gee_features.max_fire_temp(), projection, scale
        ).select('T21').mask().gt(0)
        fire_detected_data = extract_data_array(firms_binary, region, 'T21', default_value=0)
        feature_arrays['Fire Detected'] = np.flipud(fire_detected_data)

    # SWE + SWE/precip/temp trends (GEE track: Daymet V4, no asset)
    if 'Mean Annual SWE' in feature_names:
        swe = load_data(gee_features.mean_annual_swe(), projection, scale)
        swe_data = extract_data_array(swe, region, 'swe', default_value)
        feature_arrays['Mean Annual SWE'] = np.flipud(swe_data)

    if 'Trend in SWE' in feature_names:
        swe_trend = load_data(gee_features.swe_trend(), projection, scale)
        swe_trend_data = extract_data_array(swe_trend, region, 'scale', default_value)
        feature_arrays['Trend in SWE'] = np.flipud(swe_trend_data)

    if 'Trend in temperature' in feature_names:
        temp_trend = load_data(gee_features.temp_trend(), projection, scale)
        temp_trend_data = extract_data_array(temp_trend, region, 'scale', default_value)
        feature_arrays['Trend in temperature'] = np.flipud(temp_trend_data)

    if 'Trend in precipitation' in feature_names:
        precip_trend = load_data(gee_features.precip_trend(), projection, scale)
        precip_trend_data = extract_data_array(precip_trend, region, 'scale', default_value)
        feature_arrays['Trend in precipitation'] = np.flipud(precip_trend_data)

    # Projected climate change (LOCAL track): SNAP 2090s minus 2010s at cell centres.
    if 'Projected precipitation change' in feature_names:
        early = sample_local(local_rasters.SNAP_PRECIP[2010])
        late = sample_local(local_rasters.SNAP_PRECIP[2090])
        feature_arrays['Projected precipitation change'] = np.flipud(late - early)

    if 'Projected summer temperature change' in feature_names:
        early = sample_local(local_rasters.SNAP_SUMMER[2010])
        late = sample_local(local_rasters.SNAP_SUMMER[2090])
        feature_arrays['Projected summer temperature change'] = np.flipud(late - early)

    if 'Projected winter temperature change' in feature_names:
        early = sample_local(local_rasters.SNAP_WINTER[2010])
        late = sample_local(local_rasters.SNAP_WINTER[2090])
        feature_arrays['Projected winter temperature change'] = np.flipud(late - early)

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
        73: 'Lichens',
        74: 'Moss',
        81: 'Pasture/Hay',
        82: 'Cultivated Crops',
        90: 'Woody Wetlands',
        95: 'Emergent Herbaceous Wetlands'
    }
    
    if any('Land Cover' in name for name in feature_names):
        # LOCAL track: NLCD 2016 nearest-sampled at cell centres, already in the
        # flipped orientation of the other features. NaN (off-footprint) cells
        # equal no code, so they get an all-zero one-hot (the dropped 'NaN' bucket).
        landcover_array = np.flipud(sample_local(local_rasters.NLCD_IMG))

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
        # LOCAL track: ALFRESCO vegetation mode nearest-sampled at cell centres.
        vegetation_array = np.flipud(sample_local(local_rasters.VEGMODE_TIF))

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

def plot_field(field, feature_names=None, feature_stack=None, ds=None, default_value=-9999, figsize=(10, 8)):
    """Plot a field for debugging. Can use feature name, index, or numpy array directly.
    
    Args:
        field: Feature name (str), feature index (int), or 2D numpy array
        feature_names: List of feature names (required if field is name or index)
        feature_stack: 3D numpy array (height, width, features) (required if field is name or index)
        ds: xarray Dataset (alternative to feature_stack)
        default_value: Value to mask out in visualization
        figsize: Figure size tuple
    """
    # Extract 2D array based on input type
    if isinstance(field, np.ndarray):
        data = field
        title = "Field"
    elif isinstance(field, str):
        if ds is not None:
            data = ds['feature_stack'].sel(feature=field).values
            title = field
        elif feature_stack is not None and feature_names is not None:
            idx = feature_names.index(field)
            data = feature_stack[:, :, idx]
            title = field
        else:
            raise ValueError("Need feature_stack+feature_names or ds to use feature name")
    elif isinstance(field, int):
        if ds is not None:
            data = ds['feature_stack'].isel(feature=field).values
            title = ds['feature_names'].values[field]
        elif feature_stack is not None and feature_names is not None:
            data = feature_stack[:, :, field]
            title = feature_names[field]
        else:
            raise ValueError("Need feature_stack+feature_names or ds to use feature index")
    else:
        raise ValueError("field must be numpy array, feature name (str), or feature index (int)")
    
    # Mask default values for better visualization
    masked_data = np.ma.masked_where(data == default_value, data)
    
    # Print stats
    valid_data = data[data != default_value]
    print(f"\n{title}")
    print(f"  Shape: {data.shape}")
    print(f"  Valid pixels: {valid_data.size:,} ({100*valid_data.size/data.size:.1f}%)")
    if valid_data.size > 0:
        print(f"  Min: {valid_data.min():.4f}")
        print(f"  Max: {valid_data.max():.4f}")
        print(f"  Mean: {valid_data.mean():.4f}")
        print(f"  Std: {valid_data.std():.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked_data, cmap='viridis', interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# Load all features in model order and create feature stack
print("\nLoading all features for prediction...")
feature_stack = load_all_features(feature_names, SCALE, ee_roi, default_value=-9999)

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
        'description': 'Feature stack for abrupt thaw prediction model',
        'num_features': len(feature_names),
        'shape': f"{feature_stack.shape[0]} x {feature_stack.shape[1]} x {feature_stack.shape[2]}"
    }
)

# Add feature names as a coordinate variable for easy access
ds['feature_names'] = ('feature', feature_names)

# Save to NetCDF
feature_stack_path = data / 'prediction_data.nc'
ds.to_netcdf(feature_stack_path)
print(f"\nFeature stack and metadata saved to: {feature_stack_path}")
print(f"  Shape: {feature_stack.shape}")
print(f"  Features: {len(feature_names)}")
print(f"  Scale: {SCALE}m")
