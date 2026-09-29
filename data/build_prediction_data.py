"""Build the statewide datacube of model predictors.

Uses the same two tracks as build_feature_table.py: GEE layers (including
``gee_features.py``) and local rasters sampled at the datacube's cell centres via
``local_rasters.py``.
"""

import ee

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import EE_PROJECT

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)

import json
import math
import numpy as np
import xgboost as xgb
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA, MODELS
import gee_features
import local_rasters
import ee_sampling

data = DATA

# roi.geojson is the Alaska land boundary clipped to the mainland bbox
# [-170,-141]x[51,72], so there is no antimeridian wrap. The Obu mask trims it to
# the permafrost domain downstream.
with open(data / 'roi.geojson', 'r') as f:
    roi_json = json.load(f)
ee_roi = ee.Geometry(roi_json['features'][0]['geometry'])

# 1 km matches the native scale of WorldClim/Daymet and the Obu mask.
SCALE = 1000

def load_data(image: ee.Image, projection, scale: float) -> ee.Image:
    """Load and rasterize a dataset."""
    return image.reproject(projection, scale=scale).clip(ee_roi)

def extract_data_array(
    image: ee.Image,
    region: ee.Geometry,
    band_name: str = None,
    default_value: float = None,
    max_pixels: int = 262000,
) -> np.ndarray:
    """Extract a reprojected image (from load_data) over ``region`` as a 2-D array.

    ``sampleRectangle`` caps a request at 262,144 pixels, so the array is pulled in
    horizontal strips indexed off the image's own transform, making each strip an
    exact slice of one grid. Each strip is inset a quarter-pixel so the
    cover-the-region rounding returns exactly its rows. Rows run north to south.
    """
    img = image.unmask(default_value) if default_value is not None else image
    if band_name is None:
        band_name = image.bandNames().getInfo()[0]
    band_img = img.select(band_name)

    info = band_img.projection().getInfo()
    crs = info['crs']
    a, b, c, d, e, f = info['transform']  # x = a*col + b*row + c ; y = d*col + e*row + f
    assert b == 0 and d == 0, f"sheared transform unsupported: {info['transform']}"

    ring = ee.Geometry(region).bounds().coordinates().getInfo()[0]
    xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
    col0 = math.floor((min(xs) - c) / a); col1 = math.ceil((max(xs) - c) / a)
    r_a = (max(ys) - f) / e; r_b = (min(ys) - f) / e
    row0 = math.floor(min(r_a, r_b)); row1 = math.ceil(max(r_a, r_b))
    ncols = col1 - col0; nrows = row1 - row0

    ix, iy = abs(a) * 0.25, abs(e) * 0.25  # quarter-pixel inset -> no seam overlap/gap
    x_lo = c + a * col0 + ix; x_hi = c + a * col1 - ix
    rows_per = max(1, max_pixels // max(1, ncols))

    strips = []
    for rs in range(row0, row1, rows_per):
        re_ = min(rs + rows_per, row1)
        y_a = f + e * rs; y_b = f + e * re_
        y_lo = min(y_a, y_b) + iy; y_hi = max(y_a, y_b) - iy
        rect = ee.Geometry.Rectangle(
            [min(x_lo, x_hi), y_lo, max(x_lo, x_hi), y_hi], proj=crs, geodesic=False)
        s = band_img.sampleRectangle(region=rect, defaultValue=default_value)
        arr = np.array(s.get(band_name).getInfo(), dtype=float)
        assert arr.shape == (re_ - rs, ncols), (
            f"strip rows {rs}:{re_} got {arr.shape}, expected {(re_ - rs, ncols)}")
        strips.append(arr)

    out = np.concatenate(strips, axis=0)
    assert out.shape == (nrows, ncols), f"stitched {out.shape} != {(nrows, ncols)}"
    return out

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

    # Per-cell lon/lat in GEE-native orientation; sampled arrays are flipped to
    # match the GEE features below.
    lonlat = load_data(ee.Image.pixelLonLat(), projection, scale)
    lon2d = extract_data_array(lonlat, region, 'longitude', default_value)
    lat2d = extract_data_array(lonlat, region, 'latitude', default_value)

    def sample_local(path, band=1):
        """Nearest-sample a local raster band onto the datacube grid (unflipped)."""
        flat = local_rasters.sample_points(path, lon2d.ravel(), lat2d.ravel(), band=band)
        return flat.reshape(lon2d.shape)

    def assert_local_orientation(sample, layer_name):
        """Assert a flipped LOCAL categorical layer is not vertically mirrored.

        A correctly oriented layer has data where the Flammability reference does,
        not where its mirror does. Skips when the footprint is ~full (NLCD has no
        nodata); Land Cover shares Vegetation Mode's code path, so Vegetation Mode
        covers it.
        """
        ref_fp = np.isfinite(np.flipud(sample_local(local_rasters.FLAMMABILITY_TIF)))
        if not (ref_fp.any() and (~ref_fp).any()):
            return
        cat_fp = np.isfinite(sample)
        if cat_fp.mean() > 0.98:
            return
        oriented = cat_fp[ref_fp].mean()
        mirror = cat_fp[np.flipud(ref_fp)].mean()
        assert oriented > mirror, (
            f"{layer_name}: LOCAL categorical appears vertically mirrored "
            f"against the stack (footprint agreement {oriented:.3f} <= mirror "
            f"{mirror:.3f}) — check for a reintroduced double np.flipud (T31)."
        )

    # Native-scale sampling: collect -> sample -> distribute.
    # A 1 km reproject pyramid-aggregates features with finer native grids: terrain
    # derivatives are recomputed on the aggregated DEM, and averaging biases
    # heavy-tailed MERIT and SoilGrids values. These instead read the native pixel at
    # each cell centre, as build_feature_table.py does per training point. Bands are
    # merged into one multiband image per native scale and sampled in grid tiles.
    # Results are unflipped; each distribution site below flips once.
    #
    # Curvature (2 km) is served by reproject: its analysis grid is already 1 km.

    SOIL_SCALE = 250
    soil_vars = ['Soil Organic Carbon', 'Nitrogen', 'Bulk Density', 'Sand', 'Silt', 'Clay']
    soil_depths = {
        '0-5 cm': ('soc_0-5cm_mean', 'nitrogen_0-5cm_mean', 'bdod_0-5cm_mean', 'sand_0-5cm_mean', 'silt_0-5cm_mean', 'clay_0-5cm_mean'),
        '5-15 cm': ('soc_5-15cm_mean', 'nitrogen_5-15cm_mean', 'bdod_5-15cm_mean', 'sand_5-15cm_mean', 'silt_5-15cm_mean', 'clay_5-15cm_mean'),
        '15-30 cm': ('soc_15-30cm_mean', 'nitrogen_15-30cm_mean', 'bdod_15-30cm_mean', 'sand_15-30cm_mean', 'silt_15-30cm_mean', 'clay_15-30cm_mean'),
        '30-60 cm': ('soc_30-60cm_mean', 'nitrogen_30-60cm_mean', 'bdod_30-60cm_mean', 'sand_30-60cm_mean', 'silt_30-60cm_mean', 'clay_30-60cm_mean'),
        '60-100 cm': ('soc_60-100cm_mean', 'nitrogen_60-100cm_mean', 'bdod_60-100cm_mean', 'sand_60-100cm_mean', 'silt_60-100cm_mean', 'clay_60-100cm_mean'),
        '100-200 cm': ('soc_100-200cm_mean', 'nitrogen_100-200cm_mean', 'bdod_100-200cm_mean', 'sand_100-200cm_mean', 'silt_100-200cm_mean', 'clay_100-200cm_mean'),
    }
    soil_images = {
        'Soil Organic Carbon': ee.Image('projects/soilgrids-isric/soc_mean'),
        'Nitrogen': ee.Image('projects/soilgrids-isric/nitrogen_mean'),
        'Bulk Density': ee.Image('projects/soilgrids-isric/bdod_mean'),
        'Sand': ee.Image('projects/soilgrids-isric/sand_mean'),
        'Silt': ee.Image('projects/soilgrids-isric/silt_mean'),
        'Clay': ee.Image('projects/soilgrids-isric/clay_mean'),
    }
    _soil_depth_ranges = {'0-30 cm': ('0-5 cm', '5-15 cm', '15-30 cm'),
                          '30-200 cm': ('30-60 cm', '60-100 cm', '100-200 cm')}

    # Slope is needed both as a feature and for the aspect flats mask.
    _need_slope = ('Slope' in feature_names
                   or any(n in feature_names for n in ('Northness', 'Eastness')))

    # These guards must mirror the distribution sites below.
    native_req = {}
    def _req(scale_native, key, single_band_image):
        native_req.setdefault(scale_native, {})[key] = single_band_image

    if 'Elevation' in feature_names:
        _req(10, 'elevation', elevation_image)
    if _need_slope:
        _req(10, 'slope', ee.Terrain.slope(elevation_image))
    if any(n in feature_names for n in ('Northness', 'Eastness')):
        _req(10, 'aspect', ee.Terrain.aspect(elevation_image))
    if 'Mean curvature (500 m)' in feature_names:
        _req(250, 'MeanCurvature', gee_features.mean_curvature(500).select('MeanCurvature'))
    if 'Height Above Nearest Drainage' in feature_names:
        _req(gee_features.MERIT_SCALE, 'hnd', gee_features.height_above_drainage())
    if 'Upstream Area' in feature_names:
        _req(gee_features.MERIT_SCALE, 'upa', gee_features.upstream_area())
    for _var in soil_vars:
        _var_idx = soil_vars.index(_var)
        for _drange, _dbands in _soil_depth_ranges.items():
            if f'{_var} ({_drange})' in feature_names:
                for _depth in _dbands:
                    _band = soil_depths[_depth][_var_idx]
                    _req(SOIL_SCALE, _band, soil_images[_var].select(_band))

    native = {}
    for _scale in sorted(native_req):
        _keys = list(native_req[_scale])
        _multi = native_req[_scale][_keys[0]].rename(_keys[0])
        for _k in _keys[1:]:
            _multi = _multi.addBands(native_req[_scale][_k].rename(_k))
        print(f"Native-sampling {len(_keys)} band(s) @ {_scale} m: {_keys}", flush=True)

        def _log(done, total, _s=_scale):
            if done == total or done % 25 == 0:
                print(f"  [native {_s} m] tiles {done}/{total}", flush=True)

        native.update(ee_sampling.sample_native_multiband_tiled(
            lon2d, lat2d, _multi, _keys, _scale, tile=128, workers=8, log=_log))

    if 'Elevation' in feature_names:
        feature_arrays['Elevation'] = np.flipud(native['elevation'])

    slope_deg = np.flipud(native['slope']) if _need_slope else None
    if 'Slope' in feature_names:
        feature_arrays['Slope'] = slope_deg

    # Flats (slope < 1 deg) are set to 0, as in build_feature_table.py.
    if any(n in feature_names for n in ('Northness', 'Eastness')):
        aspect_deg = np.flipud(native['aspect'])
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

    if 'Mean curvature (500 m)' in feature_names:
        feature_arrays['Mean curvature (500 m)'] = np.flipud(native['MeanCurvature'])

    if 'Mean curvature (2 km)' in feature_names:
        curve2k = load_data(gee_features.mean_curvature(2000).select('MeanCurvature'), projection, scale)
        curve2k_data = extract_data_array(curve2k, region, 'MeanCurvature', default_value)
        feature_arrays['Mean curvature (2 km)'] = np.flipud(curve2k_data)

    if 'Height Above Nearest Drainage' in feature_names:
        feature_arrays['Height Above Nearest Drainage'] = np.flipud(native['hnd'])
    if 'Upstream Area' in feature_names:
        feature_arrays['Upstream Area'] = np.flipud(native['upa'])

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
    
    if 'Flammability Index' in feature_names:
        feature_arrays['Flammability Index'] = np.flipud(sample_local(local_rasters.FLAMMABILITY_TIF))

    for _feat, _band in local_rasters.MODIS_FIRE_BANDS.items():
        if _feat in feature_names:
            feature_arrays[_feat] = np.flipud(sample_local(local_rasters.MODIS_FIRE_TIF, _band))

    for _feat, _band in local_rasters.DAYMET_BANDS.items():
        if _feat in feature_names:
            feature_arrays[_feat] = np.flipud(sample_local(local_rasters.DAYMET_TIF, _band))

    # NaN off-ROI, from the -9999 lon/lat fill.
    if 'Yedoma' in feature_names:
        yedoma_flat = local_rasters.sample_yedoma(lon2d.ravel(), lat2d.ravel())
        feature_arrays['Yedoma'] = np.flipud(yedoma_flat.reshape(lon2d.shape))

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
        # NaN cells match no code, so they get an all-zero one-hot.
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
        vegetation_array = np.flipud(sample_local(local_rasters.VEGMODE_TIF))
        assert_local_orientation(vegetation_array, 'Vegetation Mode')

        for code, label in vegetation_mode_labels.items():
            feature_name = f'Vegetation Mode ({label})'
            if feature_name in feature_names:
                feature_arrays[feature_name] = (vegetation_array == code).astype(float)
    
    # Depth-weighted mean of the native-sampled soil bands.
    _soil_range_weights = {'0-30 cm': (('0-5 cm', 5), ('5-15 cm', 10), ('15-30 cm', 15)),
                           '30-200 cm': (('30-60 cm', 30), ('60-100 cm', 40), ('100-200 cm', 100))}
    _soil_range_total = {'0-30 cm': 30, '30-200 cm': 170}
    for var in soil_vars:
        var_idx = soil_vars.index(var)
        for depth_range, dbw in _soil_range_weights.items():
            feature_name = f'{var} ({depth_range})'
            if feature_name in feature_names:
                arrays = []
                for depth, weight in dbw:
                    band = soil_depths[depth][var_idx]
                    arrays.append(np.flipud(native[band]) * weight)
                feature_arrays[feature_name] = sum(arrays) / _soil_range_total[depth_range]

    
    feature_stack = np.stack([feature_arrays[name] for name in feature_names], axis=-1)
    # Off-ROI lon/lat keep the -9999 fill, which sample_points reads as NaN.
    return feature_stack, np.flipud(lon2d), np.flipud(lat2d)

def main():
    print("\nLoading all features for prediction...")
    feature_stack, lon2d, lat2d = load_all_features(feature_names, SCALE, ee_roi, default_value=-9999)

    print(f"\nFeature stack shape: {feature_stack.shape}")
    print(f"Expected shape: (height, width, {len(feature_names)})")

    ds = xr.Dataset(
        {
            'feature_stack': (['y', 'x', 'feature'], feature_stack)
        },
        coords={
            'feature': feature_names,
            'x': np.arange(feature_stack.shape[1]),
            'y': np.arange(feature_stack.shape[0]),
            'longitude': (['y', 'x'], lon2d),
            'latitude': (['y', 'x'], lat2d),
        },
        attrs={
            'scale': SCALE,
            'default_value': -9999,
            'description': 'Feature stack for abrupt thaw prediction model',
            'num_features': len(feature_names),
            'shape': f"{feature_stack.shape[0]} x {feature_stack.shape[1]} x {feature_stack.shape[2]}"
        }
    )

    ds['feature_names'] = ('feature', feature_names)

    feature_stack_path = data / 'prediction_data.nc'
    ds.to_netcdf(feature_stack_path)
    print(f"\nFeature stack and metadata saved to: {feature_stack_path}")
    print(f"  Shape: {feature_stack.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Scale: {SCALE}m")


if __name__ == '__main__':
    main()
