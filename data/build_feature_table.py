"""Build a feature table for the thaw database.

Two-track feature sourcing (no custom GEE assets; see TASKS T0):
  * GEE track   -> public catalog data sampled server-side. Public datasets
    (3DEP terrain, WorldClim bioclim, SoilGrids) are sampled inline here, as are
    the re-derived curvature and MERIT Hydro layers (``gee_features.py``).
  * LOCAL track -> ``local_rasters.py`` nearest-samples downloaded rasters at the
    point coordinates: ALFRESCO flammability + vegetation mode, NLCD land cover,
    Yedoma presence, the Daymet SWE + SWE/precip/temp trends, and the MODIS
    MCD64A1 fire history — the last two materialized to local rasters
    (``build_daymet_rasters.py`` / ``build_modis_fire_rasters.py``) because their
    deep temporal reductions hang if sampled live at scattered points (T30).
There is no ``ASSET_ROOT`` dependency.

Robustness (T39)
----------------
The full run is ~12 h, so it must never lose work:
  * Every feature is added through ``try_add`` -> a per-feature failure is
    recorded (``failed_features``) and printed, never aborting the build
    (continue-on-failure). This covers BOTH tracks (GEE and LOCAL), so a missing
    local raster or a bad band drops one column instead of discarding hours of
    unrelated work.
  * The end-of-run report + ``features_dirty.csv`` write run in a ``finally`` so
    they execute even if something structural raises. The report loudly names
    every feature that raised (``failed_features``) or came back entirely empty
    (all-NaN), so an incomplete table can never pass silently downstream.
  * Initialization is non-interactive-safe: a valid cached token never triggers a
    browser prompt that would hang an unattended run; ``ee.Authenticate()`` is a
    fallback only.
"""

import ee

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import EE_PROJECT

# Non-interactive-safe init (T39): use cached credentials first so a valid token
# never opens a browser prompt that would hang the overnight run. Only fall back
# to the interactive ee.Authenticate() flow if initialization actually fails.
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)


import os
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA
import gee_features
import local_rasters

data = DATA
# Output path is overridable so a smoke run can write elsewhere (see below).
OUT = Path(os.environ.get('FEATURE_BUILD_OUT', data / 'features_dirty.csv'))

thawdb = pd.read_csv(data / 'Alaska_Permafrost_Thaw_Database_v2.0.0.csv', sep = ',', encoding = 'latin1')

# Fast self-test hook (T39): FEATURE_BUILD_LIMIT=<N> subsets to N evenly-spaced
# points so the ENTIRE script — every feature block, the LOCAL track, the aspect
# encoding, the report, and the CSV write — runs end-to-end in minutes before the
# real ~12 h run is launched. Unset (the overnight run) processes all points.
_limit = os.environ.get('FEATURE_BUILD_LIMIT')
if _limit:
    _idx = np.linspace(0, len(thawdb) - 1, int(_limit)).astype(int)
    thawdb = thawdb.iloc[_idx].reset_index(drop=True).copy()
    print(f'[FEATURE_BUILD_LIMIT] self-test on {len(thawdb)} evenly-spaced points -> {OUT}')

print(thawdb['ThawType'].value_counts()) # 6.79% non-abrupt, 93.21% abrupt (v2.0.0)
thawdb['Class'] = np.where(thawdb['ThawType'] == 'Abrupt', 0, 1) # ABRUPT = 0 (majority class), GRADUAL = 1 (minority class)

def sample_raster(
    image: ee.Image,
    feat: ee.Feature,
    reducer: ee.Reducer,
    scale: float,
    crs: str = 'EPSG:4326'
) -> float:
    """Sample a raster at the point corresponding to a single feature."""
    point = feat.geometry(proj = crs)
    value = image.reduceRegion(
        reducer = reducer,
        geometry = point,
        scale = scale,
        crs = crs
    )
    return feat.set(value)

def add_feature(
    df: pd.DataFrame,
    points: ee.FeatureCollection,
    image: ee.Image,
    reducer: ee.Reducer,
    scale: float,
    name: str,
    band: str,
    crs: str = 'EPSG:4326'
):
    """Append a new feature to the feature table."""
    sampler = lambda feat: sample_raster(image, feat, reducer, scale, crs)
    values = points.map(sampler)
    data = ee.data.computeFeatures({
        'expression': values,
        'fileFormat': 'PANDAS_DATAFRAME'
    })
    df[name] = [row[band] for idx, row in data.iterrows()]

# Create point collection for all data points
points = [ee.Feature(ee.Geometry.Point([lon, lat])) for lon, lat in zip(thawdb['Longitude'], thawdb['Latitude'])]
point_collection = ee.FeatureCollection(points)

# T30: coordinate arrays shared by the LOCAL track, plus a collector of features
# that fail to import so the end-of-run report can surface them (a timeout, a
# missing raster, or a bad band must never be swallowed).
lons, lats = thawdb['Longitude'].to_numpy(), thawdb['Latitude'].to_numpy()
failed_features = []


def try_add(name, fn):
    """Run ``fn()`` (which appends exactly one feature column) without ever
    aborting the build. Any exception is recorded in ``failed_features`` and
    printed, so the end-of-run report names every feature that raised (T39
    continue-on-failure). Applies to BOTH tracks."""
    try:
        fn()
        print('Added', name)
    except Exception as e:
        failed_features.append((name, repr(e)))
        print('Could not add', name, ':', repr(e))


def try_add_col(name, compute):
    """Guarded LOCAL-track add: ``compute()`` returns the column values and the
    assignment is a direct subscript set (``thawdb[name] = ...``). This avoids
    ``thawdb.__setitem__(...)`` inside a closure, which materializes a bound-method
    reference that trips pandas' chained-assignment FutureWarning (harmless in
    pandas 2.x, an error in 3.0)."""
    def _assign():
        thawdb[name] = compute()
    try_add(name, _assign)


def finalize():
    """Always-run end-of-run report + save (T39 crash-safety). Runs from a
    ``finally`` so it executes even if a later step raised, meaning an overnight
    run never loses the hours of completed feature work. Loudly names every
    feature that raised (``failed_features``) or came through entirely empty
    (all-NaN) so an incomplete table cannot pass silently downstream."""
    all_nan = [c for c in thawdb.columns
               if np.issubdtype(thawdb[c].dtype, np.number) and thawdb[c].isna().all()]

    print('\n' + '=' * 70)
    print('[T30/T39] Feature import report')
    print('=' * 70)
    if failed_features:
        print('Features that raised during import (missing from the table):')
        for fname, err in failed_features:
            print(f'  - {fname}: {err}')
    if all_nan:
        print('Columns present but ENTIRELY empty (all-NaN) — check these:')
        for c in all_nan:
            print(f'  - {c}')
    if not failed_features and not all_nan:
        print('All features imported with at least some valid values.')
    print('=' * 70 + '\n')

    print(thawdb.columns)
    print(thawdb.shape)
    thawdb.to_csv(OUT, index=False)
    print(f'wrote {OUT}')


# Everything that populates the table runs under one try/finally so the report +
# save always execute, even on an unhandled error partway through.
try:
    # VARIABLE: Land cover -> LOCAL track (see the LOCAL block near the end).

    # VARIABLES: terrain analysis
    # T37: terrain is sampled at NATIVE scale (10 m / 250 m / 1000 m) on purpose —
    # do not "fix" this to 4 km / 1 km. The probe (diagnostics/probe_native_serve.py)
    # showed a coarse reproject pyramid-aggregates the derivative (slope collapses to
    # ~0.28x native at 4 km); the datacube matches this native sampling by
    # point-sampling its 1 km cell centres, so train and serve agree at native scale.
    # Each terrain call reconstructs its own 3DEP reference (a lazy ee.Image, no
    # network) so no shared variable can break a sibling feature (T39).
    _DEM = 'USGS/3DEP/10m'
    try_add('Elevation', lambda: add_feature(
        thawdb, point_collection, ee.Image(_DEM).select('elevation'),
        ee.Reducer.mean(), 10, 'Elevation', 'elevation'))
    try_add('Slope', lambda: add_feature(
        thawdb, point_collection, ee.Terrain.slope(ee.Image(_DEM).select('elevation')),
        ee.Reducer.mean(), 10, 'Slope', 'slope'))
    # T32: sample raw aspect (native) into a temporary column, then encode as
    # northness/eastness and neutralize flats below; raw circular Aspect is dropped
    # and never enters the model set.
    try_add('Aspect', lambda: add_feature(
        thawdb, point_collection, ee.Terrain.aspect(ee.Image(_DEM).select('elevation')),
        ee.Reducer.mean(), 10, 'Aspect', 'aspect'))

    # GEE track: curvature re-derived inline from 3DEP via the TAGEE-family port
    # (gee_features.mean_curvature), no custom asset. Sampled at the analysis cell
    # size (window/2 = 250 m / 1000 m).
    try_add('Mean curvature (500 m)', lambda: add_feature(
        thawdb, point_collection, gee_features.mean_curvature(500),
        ee.Reducer.mean(), 250, 'Mean curvature (500 m)', 'MeanCurvature'))
    try_add('Mean curvature (2 km)', lambda: add_feature(
        thawdb, point_collection, gee_features.mean_curvature(2000),
        ee.Reducer.mean(), 1000, 'Mean curvature (2 km)', 'MeanCurvature'))

    # VARIABLES: hydrological terrain (MERIT Hydro v1.0.1, T34)
    # GEE track, MERIT/Hydro/v1_0_1 (official catalog, NOT the sat-io mirror); native
    # ~90 m. Both features are sampled at native scale here — a point sample reads one
    # native pixel, so there is no aggregation-order concern in this path. Both `hnd`
    # and `upa` are raw and served natively in the datacube too (like the 3DEP terrain,
    # T37), so no reproject-averaging occurs and the canonical set stays raw/physical;
    # the T13 linear baseline logs `upa` in its own scope (T35). See gee_features docstrings.
    try_add('Height Above Nearest Drainage', lambda: add_feature(
        thawdb, point_collection, gee_features.height_above_drainage(),
        ee.Reducer.mean(), gee_features.MERIT_SCALE, 'Height Above Nearest Drainage', 'hnd'))
    try_add('Upstream Area', lambda: add_feature(
        thawdb, point_collection, gee_features.upstream_area(),
        ee.Reducer.mean(), gee_features.MERIT_SCALE, 'Upstream Area', 'upa'))

    # VARIABLES: bioclimatic variables
    bioclim = ee.Image('WORLDCLIM/V1/BIO')
    biovars = {
        'bio01': 'Annual Mean Temperature',
        'bio02': 'Mean Diurnal Range',
        'bio03': 'Isothermality',
        'bio04': 'Temperature Seasonality',
        'bio05': 'Max Temperature of Warmest Month',
        'bio06': 'Min Temperature of Coldest Month',
        'bio07': 'Temperature Annual Range',
        'bio08': 'Mean Temperature of Wettest Quarter',
        'bio09': 'Mean Temperature of Driest Quarter',
        'bio10': 'Mean Temperature of Warmest Quarter',
        'bio11': 'Mean Temperature of Coldest Quarter',
        'bio12': 'Annual Precipitation',
        'bio13': 'Precipitation of Wettest Month',
        'bio14': 'Precipitation of Driest Month',
        'bio15': 'Precipitation Seasonality',
        'bio16': 'Precipitation of Wettest Quarter',
        'bio17': 'Precipitation of Driest Quarter',
        'bio18': 'Precipitation of Warmest Quarter',
        'bio19': 'Precipitation of Coldest Quarter'
    }
    for band, name in biovars.items():
        try_add(name, lambda b=band, n=name: add_feature(
            thawdb, point_collection, bioclim, ee.Reducer.mean(), 1000, n, b))

    # VARIABLES: soil texture, nitrogen, organic carbon (SoilGrids, native 250 m).
    soil_sources = {
        'projects/soilgrids-isric/soc_mean': {
            'soc_0-5cm_mean': 'Soil Organic Carbon (0-5 cm)',
            'soc_5-15cm_mean': 'Soil Organic Carbon (5-15 cm)',
            'soc_15-30cm_mean': 'Soil Organic Carbon (15-30 cm)',
            'soc_30-60cm_mean': 'Soil Organic Carbon (30-60 cm)',
            'soc_60-100cm_mean': 'Soil Organic Carbon (60-100 cm)',
            'soc_100-200cm_mean': 'Soil Organic Carbon (100-200 cm)',
        },
        'projects/soilgrids-isric/nitrogen_mean': {
            'nitrogen_0-5cm_mean': 'Nitrogen (0-5 cm)',
            'nitrogen_5-15cm_mean': 'Nitrogen (5-15 cm)',
            'nitrogen_15-30cm_mean': 'Nitrogen (15-30 cm)',
            'nitrogen_30-60cm_mean': 'Nitrogen (30-60 cm)',
            'nitrogen_60-100cm_mean': 'Nitrogen (60-100 cm)',
            'nitrogen_100-200cm_mean': 'Nitrogen (100-200 cm)',
        },
        'projects/soilgrids-isric/clay_mean': {
            'clay_0-5cm_mean': 'Clay (0-5 cm)',
            'clay_5-15cm_mean': 'Clay (5-15 cm)',
            'clay_15-30cm_mean': 'Clay (15-30 cm)',
            'clay_30-60cm_mean': 'Clay (30-60 cm)',
            'clay_60-100cm_mean': 'Clay (60-100 cm)',
            'clay_100-200cm_mean': 'Clay (100-200 cm)',
        },
        'projects/soilgrids-isric/sand_mean': {
            'sand_0-5cm_mean': 'Sand (0-5 cm)',
            'sand_5-15cm_mean': 'Sand (5-15 cm)',
            'sand_15-30cm_mean': 'Sand (15-30 cm)',
            'sand_30-60cm_mean': 'Sand (30-60 cm)',
            'sand_60-100cm_mean': 'Sand (60-100 cm)',
            'sand_100-200cm_mean': 'Sand (100-200 cm)',
        },
        # Silt is still sampled at source; clean_feature_table.py drops it (T35,
        # closed sand/silt/clay composition). Kept here so the dirty table is complete.
        'projects/soilgrids-isric/silt_mean': {
            'silt_0-5cm_mean': 'Silt (0-5 cm)',
            'silt_5-15cm_mean': 'Silt (5-15 cm)',
            'silt_15-30cm_mean': 'Silt (15-30 cm)',
            'silt_30-60cm_mean': 'Silt (30-60 cm)',
            'silt_60-100cm_mean': 'Silt (60-100 cm)',
            'silt_100-200cm_mean': 'Silt (100-200 cm)',
        },
        'projects/soilgrids-isric/bdod_mean': {
            'bdod_0-5cm_mean': 'Bulk Density (0-5 cm)',
            'bdod_5-15cm_mean': 'Bulk Density (5-15 cm)',
            'bdod_15-30cm_mean': 'Bulk Density (15-30 cm)',
            'bdod_30-60cm_mean': 'Bulk Density (30-60 cm)',
            'bdod_60-100cm_mean': 'Bulk Density (60-100 cm)',
            'bdod_100-200cm_mean': 'Bulk Density (100-200 cm)',
        },
    }
    for asset, bandmap in soil_sources.items():
        img = ee.Image(asset)
        for band, name in bandmap.items():
            try_add(name, lambda i=img, b=band, n=name: add_feature(
                thawdb, point_collection, i, ee.Reducer.mean(), 250, n, b))

    # --------------------------------------------------------------------------
    # LOCAL track: nearest-sample downloaded rasters at the point coordinates for
    # the features with no GEE-catalog upstream (local_rasters.py). Land Cover and
    # Vegetation Mode stay raw integer codes here; clean_feature_table.py one-hot
    # encodes them (0 / nodata -> 'NaN' bucket, dropped there). Every LOCAL feature
    # is guarded by try_add too (T39): a missing/corrupt raster drops one column
    # rather than aborting the whole run at the very end.
    # --------------------------------------------------------------------------

    # Land cover (NLCD 2016): missing -> code 0 so clean's land_cover_labels[0]='NaN'.
    def _add_land_cover():
        lc = local_rasters.sample_points(local_rasters.NLCD_IMG, lons, lats)
        thawdb['Land Cover'] = np.where(np.isnan(lc), 0.0, lc)
    try_add('Land Cover', _add_land_cover)

    # Vegetation mode (ALFRESCO): keep NaN for nodata; clean skips the NaN category.
    try_add_col('Vegetation Mode',
                lambda: local_rasters.sample_points(local_rasters.VEGMODE_TIF, lons, lats))

    # Flammability index (ALFRESCO), continuous.
    try_add_col('Flammability Index',
                lambda: local_rasters.sample_points(local_rasters.FLAMMABILITY_TIF, lons, lats))

    # Mean annual SWE + SWE/precip/temp trends (Daymet V4): deep temporal reductions
    # that hang when point-sampled live on GEE (T30), so materialized once to a local
    # 1 km raster by build_daymet_rasters.py and read here. Bands per
    # local_rasters.DAYMET_BANDS. Each band is guarded separately.
    for _feat, _band in local_rasters.DAYMET_BANDS.items():
        try_add_col(_feat, lambda b=_band: local_rasters.sample_points(
            local_rasters.DAYMET_TIF, lons, lats, band=b))

    # Fire history (MODIS MCD64A1, T36): Time Since Last Fire + Burn Count. Like
    # Daymet, deep temporal reductions that hang when point-sampled live on GEE (T30),
    # so materialized once to a local ~500 m raster by build_modis_fire_rasters.py and
    # read here. Both are right-censored to the ~24-yr record ("no fire since 2001" !=
    # never-burned; see gee_features). Bands per local_rasters.MODIS_FIRE_BANDS.
    for _feat, _band in local_rasters.MODIS_FIRE_BANDS.items():
        try_add_col(_feat, lambda b=_band: local_rasters.sample_points(
            local_rasters.MODIS_FIRE_TIF, lons, lats, band=b))

    # Yedoma (IRYP v2, T33): binary confirmed-presence via point-in-polygon. The
    # datacube path runs the identical sample_yedoma call at its cell centres, so
    # train/serve parity is exact by construction (as with T37 terrain).
    try_add_col('Yedoma', lambda: local_rasters.sample_yedoma(lons, lats))

    # --------------------------------------------------------------------------
    # T32: encode aspect as northness = cos(aspect), eastness = sin(aspect). Raw
    # Aspect is circular (0 deg == 360 deg) and non-monotonic, which a tree splits
    # poorly; the cos/sin pair is continuous and reprojection-safe. On flats
    # (slope < 1 deg) there is no preferred direction, so both are neutralized to 0
    # (keeping the row's other terrain info). Raw Aspect is dropped from the table.
    # Guarded so a terrain-sampling failure upstream can't abort the run here.
    # --------------------------------------------------------------------------
    def _encode_aspect():
        if 'Aspect' not in thawdb.columns:
            return
        _asp = np.deg2rad(thawdb['Aspect'].to_numpy(dtype=float))
        _flat = thawdb['Slope'].to_numpy(dtype=float) < 1.0  # NaN slope -> False (kept)
        _north = np.cos(_asp)
        _east = np.sin(_asp)
        _north[_flat] = 0.0
        _east[_flat] = 0.0
        thawdb['Northness'] = _north
        thawdb['Eastness'] = _east
        thawdb.drop(columns=['Aspect'], inplace=True)
        print('Encoded aspect -> Northness/Eastness (flats < 1 deg neutralized); dropped raw Aspect (T32)')
    try:
        _encode_aspect()
    except Exception as e:
        failed_features.append(('Northness/Eastness (aspect encoding)', repr(e)))
        print('Could not encode aspect -> Northness/Eastness:', repr(e))

finally:
    # T30/T39: report + save always run, even if the try body raised, so hours of
    # completed feature work are never discarded by a late failure.
    finalize()
