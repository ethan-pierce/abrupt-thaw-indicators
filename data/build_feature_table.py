"""Build the per-point feature table for the thaw database.

Features come from two tracks:
  * GEE track: public catalog data (3DEP terrain, WorldClim bioclim, SoilGrids,
    curvature and MERIT Hydro from ``gee_features.py``) sampled server-side.
  * LOCAL track: ``local_rasters.py`` samples downloaded rasters at the points
    (ALFRESCO, NLCD, Yedoma, Daymet, MODIS fire). Daymet and MODIS fire are
    materialized locally by ``build_daymet_rasters.py`` and
    ``build_modis_fire_rasters.py`` because their deep temporal reductions hang
    when sampled live at scattered points.

The full run takes ~12 h. Each feature is added through ``try_add``, so one
failure drops one column instead of aborting the build, and the report and CSV
write run in a ``finally``.
"""

import ee

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import EE_PROJECT

# Try cached credentials first so an unattended run never blocks on a browser prompt.
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)


import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA
import gee_features
import local_rasters

data = DATA
OUT = data / 'features_dirty.csv'

thawdb = pd.read_csv(data / 'Alaska_Permafrost_Thaw_Database_v2.0.0.csv', sep = ',', encoding = 'latin1')

print(thawdb['ThawType'].value_counts()) # 6.79% non-abrupt, 93.21% abrupt (v2.0.0)
thawdb['Class'] = np.where(thawdb['ThawType'] == 'Abrupt', 0, 1) # 0 = Abrupt (majority), 1 = Non-abrupt (minority)

def sample_raster(
    image: ee.Image,
    feat: ee.Feature,
    reducer: ee.Reducer,
    scale: float,
    crs: str = 'EPSG:4326'
) -> ee.Feature:
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

lons, lats = thawdb['Longitude'].to_numpy(), thawdb['Latitude'].to_numpy()
failed_features = []


def try_add(name, fn):
    """Run ``fn()``, which adds one feature column; record any exception instead of raising."""
    try:
        fn()
        print('Added', name)
    except Exception as e:
        failed_features.append((name, repr(e)))
        print('Could not add', name, ':', repr(e))


def try_add_col(name, compute):
    """``try_add`` for a column computed by ``compute()``.

    Assigns by direct subscript; ``thawdb.__setitem__`` in a closure trips pandas'
    chained-assignment warning."""
    def _assign():
        thawdb[name] = compute()
    try_add(name, _assign)


def finalize():
    """Report features that raised or came back all-NaN, then write the table."""
    all_nan = [c for c in thawdb.columns
               if np.issubdtype(thawdb[c].dtype, np.number) and thawdb[c].isna().all()]

    print('\n' + '=' * 70)
    print('Feature import report')
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


try:
    # Terrain is sampled at native scale on purpose. A coarse reproject
    # pyramid-aggregates the derivative (slope drops to ~0.28x native at 4 km); the
    # datacube point-samples its 1 km cell centres, so train and serve agree.
    _DEM = 'USGS/3DEP/10m'
    try_add('Elevation', lambda: add_feature(
        thawdb, point_collection, ee.Image(_DEM).select('elevation'),
        ee.Reducer.mean(), 10, 'Elevation', 'elevation'))
    try_add('Slope', lambda: add_feature(
        thawdb, point_collection, ee.Terrain.slope(ee.Image(_DEM).select('elevation')),
        ee.Reducer.mean(), 10, 'Slope', 'slope'))
    # Temporary column; encoded as Northness/Eastness and dropped below.
    try_add('Aspect', lambda: add_feature(
        thawdb, point_collection, ee.Terrain.aspect(ee.Image(_DEM).select('elevation')),
        ee.Reducer.mean(), 10, 'Aspect', 'aspect'))

    # Curvature is sampled at the analysis cell size (window / 2).
    try_add('Mean curvature (500 m)', lambda: add_feature(
        thawdb, point_collection, gee_features.mean_curvature(500),
        ee.Reducer.mean(), 250, 'Mean curvature (500 m)', 'MeanCurvature'))
    try_add('Mean curvature (2 km)', lambda: add_feature(
        thawdb, point_collection, gee_features.mean_curvature(2000),
        ee.Reducer.mean(), 1000, 'Mean curvature (2 km)', 'MeanCurvature'))

    # MERIT Hydro, sampled at native ~90 m here and in the datacube.
    try_add('Height Above Nearest Drainage', lambda: add_feature(
        thawdb, point_collection, gee_features.height_above_drainage(),
        ee.Reducer.mean(), gee_features.MERIT_SCALE, 'Height Above Nearest Drainage', 'hnd'))
    try_add('Upstream Area', lambda: add_feature(
        thawdb, point_collection, gee_features.upstream_area(),
        ee.Reducer.mean(), gee_features.MERIT_SCALE, 'Upstream Area', 'upa'))

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

    # SoilGrids, native 250 m.
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
        # clean_feature_table.py drops silt (sand + silt + clay is a closed composition).
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

    # LOCAL track. Land Cover and Vegetation Mode stay raw integer codes;
    # clean_feature_table.py one-hot encodes them.

    # NLCD 2016: missing -> code 0, which clean_feature_table.py labels 'NaN'.
    def _add_land_cover():
        lc = local_rasters.sample_points(local_rasters.NLCD_IMG, lons, lats)
        thawdb['Land Cover'] = np.where(np.isnan(lc), 0.0, lc)
    try_add('Land Cover', _add_land_cover)

    # ALFRESCO vegetation mode: nodata stays NaN.
    try_add_col('Vegetation Mode',
                lambda: local_rasters.sample_points(local_rasters.VEGMODE_TIF, lons, lats))

    try_add_col('Flammability Index',
                lambda: local_rasters.sample_points(local_rasters.FLAMMABILITY_TIF, lons, lats))

    for _feat, _band in local_rasters.DAYMET_BANDS.items():
        try_add_col(_feat, lambda b=_band: local_rasters.sample_points(
            local_rasters.DAYMET_TIF, lons, lats, band=b))

    # Fire history is right-censored to the MODIS record: no fire since 2001 is not never burned.
    for _feat, _band in local_rasters.MODIS_FIRE_BANDS.items():
        try_add_col(_feat, lambda b=_band: local_rasters.sample_points(
            local_rasters.MODIS_FIRE_TIF, lons, lats, band=b))

    # IRYP v2 point-in-polygon; the datacube runs the same call at its cell centres.
    try_add_col('Yedoma', lambda: local_rasters.sample_yedoma(lons, lats))

    # Aspect is circular, which trees split poorly; encode as cos/sin. Flats
    # (slope < 1 deg) have no preferred direction, so both are set to 0.
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
        print('Encoded aspect -> Northness/Eastness (flats < 1 deg neutralized); dropped raw Aspect')
    try:
        _encode_aspect()
    except Exception as e:
        failed_features.append(('Northness/Eastness (aspect encoding)', repr(e)))
        print('Could not encode aspect -> Northness/Eastness:', repr(e))

finally:
    finalize()
