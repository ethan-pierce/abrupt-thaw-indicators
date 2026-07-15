"""Build a datacube of predictors over interior and Arctic Alaska.

Same two-track sourcing as build_feature_table.py (no custom GEE assets, see
TASKS T0): public-catalog + re-derived GEE layers (curvature, MERIT Hydro terrain)
come from ``gee_features.py``, and the LOCAL features (ALFRESCO flammability +
vegetation mode, NLCD land cover, the Daymet SWE + SWE/precip/temp trends, and the
MODIS MCD64A1 fire history — the last two materialized by
``build_daymet_rasters.py`` / ``build_modis_fire_rasters.py``) are nearest-sampled
from local rasters at the datacube's own cell centres via ``local_rasters.py``.
No ``ASSET_ROOT`` dependency.
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
import ee_sampling

data = DATA

with open(data / 'roi.geojson', 'r') as f:
    roi_json = json.load(f)
ee_roi = geemap.geojson_to_ee(roi_json)

# Prediction-surface resolution (decision 2026-07-14): upscaled from 4 km to
# 1 km. 4 km was an efficiency choice, in tension with abrupt thaw being a
# fine-scale (10s-100s m) process; 1 km resolves terrain/land-cover heterogeneity
# far better, matches the native scale of the coarsest still-meaningful features
# (WorldClim/Daymet), and is comparable to the Obu mask's resolution, at ~16x the
# 4 km cell count (~975k cells over the ROI) — tractable as a single file.
SCALE = 1000

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

    def sample_local(path, band=1):
        """Nearest-sample a local raster band onto the datacube grid (native
        orientation; caller flips to match the GEE features)."""
        flat = local_rasters.sample_points(path, lon2d.ravel(), lat2d.ravel(), band=band)
        return flat.reshape(lon2d.shape)

    def sample_native(image, band, scale_native, reducer=None):
        """Native-scale cell-centre point-sample of a GEE image (TASKS T37).

        Terrain derivatives (slope/aspect/curvature) are recomputed on a
        pyramid-aggregated DEM when reprojected to 1 km, collapsing the signal
        (slope -> ~0.28x native at 4 km; see diagnostics/probe_native_serve.py),
        so they CANNOT be served by load_data(...).reproject. Instead we read the
        native pixel at each 1 km cell centre via a chunked reduceRegions at the
        image's native scale — the identical construction build_feature_table.py
        uses per training point, so train and serve agree at native scale by
        construction. Returns native (unflipped) orientation like sample_local;
        the caller flips to match the rest of the stack.
        """
        reducer = ee.Reducer.mean() if reducer is None else reducer
        flat = ee_sampling.sample_points_reduceregions_chunked(
            lon2d.ravel(), lat2d.ravel(), image, reducer, scale_native, band,
            crs='EPSG:4326')
        return flat.reshape(lon2d.shape)

    def assert_local_orientation(sample, layer_name):
        """T31 orientation guard for LOCAL categorical layers.

        Every LOCAL layer is nearest-sampled at the same cell centres and
        flipped exactly once, so a correctly oriented categorical must carry
        data where a reference LOCAL layer (Flammability) does — and *not*
        where that reference's vertical mirror does. This trips if a double
        ``np.flipud`` is ever reintroduced (the bug T31 fixed), before it can
        silently regress a render. ``sample`` is the raw sampled array in its
        final (single-flipped) orientation.

        Skips silently when there is no orientation signal: a ``sample`` whose
        footprint is ~full (e.g. NLCD declares no nodata, so off-domain reads
        as code 0, not NaN). Land Cover and Vegetation Mode share an identical
        sample+flip code path, so Vegetation Mode's informative (ALFRESCO-
        nodata) footprint transitively witnesses Land Cover's too.
        """
        ref_fp = np.isfinite(np.flipud(sample_local(local_rasters.FLAMMABILITY_TIF)))
        if not (ref_fp.any() and (~ref_fp).any()):
            return  # reference footprint carries no orientation information
        cat_fp = np.isfinite(sample)
        if cat_fp.mean() > 0.98:
            return  # ~full footprint (e.g. NLCD) — no orientation signal here
        oriented = cat_fp[ref_fp].mean()
        mirror = cat_fp[np.flipud(ref_fp)].mean()
        assert oriented > mirror, (
            f"{layer_name}: LOCAL categorical appears vertically mirrored "
            f"against the stack (footprint agreement {oriented:.3f} <= mirror "
            f"{mirror:.3f}) — check for a reintroduced double np.flipud (T31)."
        )

    # --- Terrain (T37): native cell-centre point-sampling, NOT reproject ---
    # A 1 km reproject pyramid-aggregates the native derivative (slope/aspect at
    # 10 m, curv-500 m at 250 m), so those are point-sampled at native scale via
    # sample_native. Curv-2 km's native analysis grid IS 1 km, so a reproject
    # recovers it exactly (probe_native_serve: corr 1.000) and is kept — no need
    # to point-sample ~1e6 cells for it.
    if 'Elevation' in feature_names:
        feature_arrays['Elevation'] = np.flipud(sample_native(elevation_image, 'elevation', 10))

    # Slope is needed both as a feature and for the T32 flats mask; sample once.
    _need_slope = ('Slope' in feature_names
                   or any(n in feature_names for n in ('Northness', 'Eastness')))
    slope_deg = np.flipud(sample_native(ee.Terrain.slope(elevation_image), 'slope', 10)) if _need_slope else None
    if 'Slope' in feature_names:
        feature_arrays['Slope'] = slope_deg

    # T32: Aspect -> northness/eastness, flats (slope < 1 deg) neutralized to 0.
    if any(n in feature_names for n in ('Northness', 'Eastness')):
        aspect_deg = np.flipud(sample_native(ee.Terrain.aspect(elevation_image), 'aspect', 10))
        asp_rad = np.deg2rad(aspect_deg)
        flat = slope_deg < 1.0  # NaN slope -> False (aspect kept / stays NaN)
        if 'Northness' in feature_names:
            north = np.cos(asp_rad)
            north[flat] = 0.0
            feature_arrays['Northness'] = north
        if 'Eastness' in feature_names:
            east = np.sin(asp_rad)
            east[flat] = 0.0
            feature_arrays['Eastness'] = east

    # Curvature (GEE track: TAGEE-family port, no custom asset).
    if 'Mean curvature (500 m)' in feature_names:
        # Native analysis grid 250 m -> point-sample at native scale (T37).
        feature_arrays['Mean curvature (500 m)'] = np.flipud(
            sample_native(gee_features.mean_curvature(500).select('MeanCurvature'), 'MeanCurvature', 250))

    if 'Mean curvature (2 km)' in feature_names:
        # Native analysis grid 1000 m == 1 km serve grid -> reproject is exact.
        curve2k = load_data(gee_features.mean_curvature(2000).select('MeanCurvature'), projection, scale)
        curve2k_data = extract_data_array(curve2k, region, 'MeanCurvature', default_value)
        feature_arrays['Mean curvature (2 km)'] = np.flipud(curve2k_data)

    # --- Hydrological terrain (T34): MERIT Hydro v1.0.1, native ~90 m ---
    # hnd is served NATIVELY, like the 3DEP terrain (T37): a stored height, so a
    # 1 km reproject would average the ~120 native pixels under each cell and blur
    # the valley/slope contrast. Point-sampling at MERIT_SCALE via sample_native is
    # the identical construction the point path uses, so parity is exact.
    if 'Height Above Nearest Drainage' in feature_names:
        feature_arrays['Height Above Nearest Drainage'] = np.flipud(
            sample_native(gee_features.height_above_drainage(), 'hnd', gee_features.MERIT_SCALE))

    # upa is also served NATIVELY and RAW (T35): though heavy-tailed and finer than
    # the 1 km grid, point-sampling the native pixel at MERIT_SCALE matches the
    # point path by construction, so — like hnd — no reproject-averaging occurs and
    # the log(mean(upa)) bias that averaging would introduce never arises. No log is
    # baked in (a no-op for the XGBoost fit; the T13 linear baseline logs it in its
    # own scope). This replaces the former reduceResolution(mean)-on-log path (T34).
    if 'Upstream Area' in feature_names:
        feature_arrays['Upstream Area'] = np.flipud(
            sample_native(gee_features.upstream_area(), 'upa', gee_features.MERIT_SCALE))

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

    # Fire history (LOCAL track: MODIS MCD64A1 materialized raster,
    # build_modis_fire_rasters.py, T36). Replaces the FIRMS max-fire-temp / Fire
    # Detected pair (reverted). Like Daymet, deep temporal reductions that can't be
    # sampled live on GEE without hanging (T30), so nearest-sampled at cell centres
    # from the ~500 m raster (resampled to the 1 km serve grid here). Bands per
    # local_rasters.MODIS_FIRE_BANDS.
    for _feat, _band in local_rasters.MODIS_FIRE_BANDS.items():
        if _feat in feature_names:
            feature_arrays[_feat] = np.flipud(sample_local(local_rasters.MODIS_FIRE_TIF, _band))

    # SWE + SWE/precip/temp trends (LOCAL track: Daymet V4 materialized raster,
    # build_daymet_rasters.py). Deep temporal reductions can't be sampled live on
    # GEE without hanging (T30), so they are nearest-sampled at cell centres like
    # the other LOCAL features. Bands per local_rasters.DAYMET_BANDS.
    for _feat, _band in local_rasters.DAYMET_BANDS.items():
        if _feat in feature_names:
            feature_arrays[_feat] = np.flipud(sample_local(local_rasters.DAYMET_TIF, _band))

    # Yedoma (IRYP v2, T33): binary confirmed-presence, point-in-polygon at the
    # same cell centres the build_feature_table.py point path uses per training
    # point (identical sample_yedoma call), so train/serve parity is exact by
    # construction. NaN off-ROI (from the -9999 lon/lat fill), like the LOCAL
    # rasters; flipped once to match the rest of the stack.
    if 'Yedoma' in feature_names:
        yedoma_flat = local_rasters.sample_yedoma(lon2d.ravel(), lat2d.ravel())
        feature_arrays['Yedoma'] = np.flipud(yedoma_flat.reshape(lon2d.shape))

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
        assert_local_orientation(landcover_array, 'Land Cover')

        for code, label in land_cover_labels.items():
            feature_name = f'Land Cover ({label})'
            if feature_name in feature_names:
                feature_arrays[feature_name] = (landcover_array == code).astype(float)

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
        assert_local_orientation(vegetation_array, 'Vegetation Mode')

        for code, label in vegetation_mode_labels.items():
            feature_name = f'Vegetation Mode ({label})'
            if feature_name in feature_names:
                feature_arrays[feature_name] = (vegetation_array == code).astype(float)
    
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
    
    # Load and aggregate soil variables.
    # Served NATIVELY at SoilGrids' 250 m grid (T35): each depth band is
    # point-sampled at the 1 km cell centre via sample_native (the identical
    # construction the point path uses at scale 250), NOT reproject-averaged. This
    # keeps heavy-tailed SOC/Nitrogen from being pulled up by the ~16 native pixels
    # under each 1 km cell and makes train/serve parity exact by construction. The
    # depth-compositing below (a linear weighted mean across depths) is unchanged;
    # only the horizontal aggregation moved from a 16-px mean to the centre pixel.
    SOIL_SCALE = 250
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
                    arr = sample_native(soil_images[var].select(band), band, SOIL_SCALE)
                    # Flip before aggregating (sample_native returns native orientation)
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
