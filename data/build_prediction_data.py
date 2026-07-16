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
import math
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA, MODELS
import gee_features
import local_rasters
import ee_sampling

data = DATA

# Statewide extraction footprint (T46): roi.geojson now holds the Alaska land
# boundary (TIGER 'Alaska' geometry INTERSECT the mainland bbox [-170,-141]x[51,72],
# mainland-clipped so there is no antimeridian wrap), replacing the stale North-Slope
# polygon. It is the single statewide source of truth shared with the raster builders
# (build_daymet_rasters.py / build_modis_fire_rasters.py, which union it with the
# ThawDB point bbox) and dry_run_gee.py. The Obu domain mask (T20) does the actual
# permafrost-domain trimming inside this footprint. ee.Geometry (not a FeatureCollection)
# so extract_data_array's ee.Geometry(region) strip-tiling accepts it directly.
with open(data / 'roi.geojson', 'r') as f:
    roi_json = json.load(f)
ee_roi = ee.Geometry(roi_json['features'][0]['geometry'])

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
    default_value: float = None,
    max_pixels: int = 262000,
) -> np.ndarray:
    """Extract a reprojected image (from load_data) over ``region`` as a 2-D array.

    ``sampleRectangle`` caps a single request at 262,144 pixels (512x512); the 1 km
    serve grid (T37) is ~4M pixels over the ROI, so we pull the array in horizontal
    strips and stitch. All strips are indexed off ONE global pixel grid — the image's
    own reprojected transform — so they are exact slices of a single grid, not
    independently-snapped tiles (which could misalign by a pixel at the seams). Each
    strip's rectangle is inset a quarter-pixel so ``sampleRectangle``'s cover-the-region
    rounding returns exactly its intended rows, with no seam overlap or gap. Bit-identical
    to the intended single-call reproject serve (verified: tiled == whole, max diff 0).
    Returns north-to-south rows, matching the old single call — callers' np.flipud unchanged.
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

    # --- Native-scale sampling (T37 + T47): collect -> sample -> distribute ---
    # A 1 km reproject pyramid-aggregates any feature whose native grid is finer
    # than the serve grid: terrain derivatives (slope/aspect @10 m, curv-500 @250 m)
    # are RECOMPUTED on the aggregated DEM (slope collapses to ~0.28x native at 4 km;
    # probe_native_serve), and MERIT hnd/upa (~90 m) + SoilGrids (250 m) are
    # stored/heavy-tailed quantities that averaging would bias. So all of these read
    # the NATIVE pixel at each 1 km cell centre — the identical construction
    # build_feature_table.py uses per training point, so train/serve agree at native
    # scale by construction. Rather than one full-grid reduceRegions per band (T47:
    # index chunks span the whole state → intractable), every native-band request is
    # gathered into ONE multiband image per native scale (10 / 90 / 250 m) and sampled
    # once via sample_native_multiband_tiled (one bounded-footprint reduceRegions per
    # compact 128² grid tile, concurrent, off-ROI tiles skipped). Merging is
    # numerically identical to per-band sampling (T47 parity gate). Results are native
    # (unflipped) orientation; each distribution site below flips once.
    #
    # Curv-2 km is the exception: its native analysis grid IS 1 km, so reproject
    # recovers it exactly (probe_native_serve: corr 1.000) — no need to point-sample.

    # Soil metadata (used both to register the native requests here and to composite
    # depths below). SoilGrids' 250 m grid is finer than the 1 km serve grid, so each
    # depth band is point-sampled at its native cell (T35), NOT reproject-averaged —
    # this keeps heavy-tailed SOC/Nitrogen from being pulled up by the ~16 native
    # pixels under each 1 km cell and makes train/serve parity exact by construction.
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

    # Slope is needed both as a feature and for the T32 flats mask; sample once.
    _need_slope = ('Slope' in feature_names
                   or any(n in feature_names for n in ('Northness', 'Eastness')))

    # COLLECT: unique band key -> single-band ee.Image, grouped by native scale.
    # These guards MUST mirror the distribution sites below.
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

    # SAMPLE: one tiled, concurrent multiband pass per native scale.
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

    # DISTRIBUTE (native orientation -> single flip -> post-process) ---------------
    if 'Elevation' in feature_names:
        feature_arrays['Elevation'] = np.flipud(native['elevation'])

    slope_deg = np.flipud(native['slope']) if _need_slope else None
    if 'Slope' in feature_names:
        feature_arrays['Slope'] = slope_deg

    # T32: Aspect -> northness/eastness, flats (slope < 1 deg) neutralized to 0.
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
        # Native analysis grid 1000 m == 1 km serve grid -> reproject is exact.
        curve2k = load_data(gee_features.mean_curvature(2000).select('MeanCurvature'), projection, scale)
        curve2k_data = extract_data_array(curve2k, region, 'MeanCurvature', default_value)
        feature_arrays['Mean curvature (2 km)'] = np.flipud(curve2k_data)

    # Hydrological terrain (T34/T35): MERIT Hydro hnd/upa served natively (raw upa,
    # T35 — no log baked in; the T13 linear baseline logs it in its own scope).
    if 'Height Above Nearest Drainage' in feature_names:
        feature_arrays['Height Above Nearest Drainage'] = np.flipud(native['hnd'])
    if 'Upstream Area' in feature_names:
        feature_arrays['Upstream Area'] = np.flipud(native['upa'])

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
    
    # Composite soil depths from the native-sampled bands (collected + sampled in
    # the native block above; soil_vars/soil_depths were defined there). Each 250 m
    # depth band was point-sampled at the 1 km cell centre (T35, native orientation);
    # here we flip once and take the linear depth-weighted mean — unchanged from the
    # per-band sample_native path, only the sampling moved into the batched pass.
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
                    # native[band] is native orientation -> flip once, then weight.
                    arrays.append(np.flipud(native[band]) * weight)
                feature_arrays[feature_name] = sum(arrays) / _soil_range_total[depth_range]

    
    # Stack features in the exact order required by the model
    feature_stack = np.stack([feature_arrays[name] for name in feature_names], axis=-1)
    # Per-cell WGS84 lon/lat, flipped once to match the (y, x) orientation of the
    # flipped feature stack, so they georeference the saved datacube (T20: the mask
    # / predict.py sample Obu at these coords; T21 / map axes also consume them).
    # Off-ROI cells keep the -9999 fill, which sample_points reads as NaN.
    return feature_stack, np.flipud(lon2d), np.flipud(lat2d)

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

def main():
    # Load all features in model order and create feature stack
    print("\nLoading all features for prediction...")
    feature_stack, lon2d, lat2d = load_all_features(feature_names, SCALE, ee_roi, default_value=-9999)

    print(f"\nFeature stack shape: {feature_stack.shape}")
    print(f"Expected shape: (height, width, {len(feature_names)})")

    # Create xarray Dataset with feature stack and metadata. longitude/latitude are
    # 2D (y, x) coords that georeference every cell centre (T20): predict.py samples
    # the Obu domain mask at these coords, and they carry the -9999 off-ROI fill so
    # out-of-footprint cells resolve to NaN (masked) downstream.
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

    # Add feature names as a coordinate variable for easy access
    ds['feature_names'] = ('feature', feature_names)

    # Save to NetCDF
    feature_stack_path = data / 'prediction_data.nc'
    ds.to_netcdf(feature_stack_path)
    print(f"\nFeature stack and metadata saved to: {feature_stack_path}")
    print(f"  Shape: {feature_stack.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Scale: {SCALE}m")


if __name__ == '__main__':
    main()
