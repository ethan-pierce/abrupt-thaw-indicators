"""Build a feature table for the thaw database.

Two-track feature sourcing (no custom GEE assets; see TASKS T0):
  * GEE track   -> public catalog data sampled server-side. Public datasets
    (3DEP terrain, WorldClim bioclim, SoilGrids) are sampled inline here;
    the climate/terrain layers that were formerly custom assets (curvature,
    SWE + trends, max fire temperature) come from ``gee_features.py``.
  * LOCAL track -> ``local_rasters.py`` nearest-samples downloaded rasters at
    the point coordinates for the three features with no GEE-catalog upstream
    (ALFRESCO flammability + vegetation mode, NLCD land cover).
There is no ``ASSET_ROOT`` dependency.
"""

import ee

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import EE_PROJECT

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA
import gee_features
import local_rasters
import ee_sampling

data = DATA

thawdb = pd.read_csv(data / 'Alaska_Permafrost_Thaw_Database_v2.0.0.csv', sep = ',', encoding = 'latin1')

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

# T30: coordinate arrays shared by the LOCAL track and ee_sampling's chunked
# reduceRegions fallback, plus a collector of features that fail to import so
# the end-of-run report can surface them (a timeout must never be swallowed).
lons, lats = thawdb['Longitude'].to_numpy(), thawdb['Latitude'].to_numpy()
failed_features = []

# VARIABLE: Land cover -> LOCAL track (see the LOCAL block near the end).

# VARIABLES: terrain analysis
try:
    elevation = ee.Image('USGS/3DEP/10m').select('elevation')
    add_feature(thawdb, point_collection, elevation, ee.Reducer.mean(), 10, 'Elevation', 'elevation')
    print('Added USGS 3DEP elevation')
except:
    print('Could not add USGS 3DEP elevation')

try:
    slope = ee.Terrain.slope(elevation)
    add_feature(thawdb, point_collection, slope, ee.Reducer.mean(), 10, 'Slope', 'slope')
    print('Added slope derived from USGS 3DEP elevation')
except:
    print('Could not add slope derived from USGS 3DEP elevation')

try:
    aspect = ee.Terrain.aspect(elevation)
    add_feature(thawdb, point_collection, aspect, ee.Reducer.mean(), 10, 'Aspect', 'aspect')
    print('Added aspect derived from USGS 3DEP elevation')
except:
    print('Could not add aspect derived from USGS 3DEP elevation')

# GEE track: curvature re-derived inline from 3DEP via the TAGEE-family port
# (gee_features.mean_curvature), no custom asset. Sampled at the analysis cell
# size (window/2 = 250 m / 1000 m).
try:
    curve500 = gee_features.mean_curvature(500)
    add_feature(thawdb, point_collection, curve500, ee.Reducer.mean(), 250, 'Mean curvature (500 m)', 'MeanCurvature')
    print('Added mean 500m curvature (TAGEE-family port of USGS 3DEP elevation)')
except Exception as e:
    print('Could not add mean 500m curvature:', e)

try:
    curve2k = gee_features.mean_curvature(2000)
    add_feature(thawdb, point_collection, curve2k, ee.Reducer.mean(), 1000, 'Mean curvature (2 km)', 'MeanCurvature')
    print('Added mean 2km curvature (TAGEE-family port of USGS 3DEP elevation)')
except Exception as e:
    print('Could not add mean 2km curvature:', e)

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
    try:
        add_feature(thawdb, point_collection, bioclim, ee.Reducer.mean(), 1000, name, band)
        print('Added', name, 'from WorldClim bioclimatic variables')
    except:
        print('Could not add', name, 'from WorldClim bioclimatic variables')

# # VARIABLES: soil texture, nitrogen, organic carbon
carbon = ee.Image('projects/soilgrids-isric/soc_mean')
bandmap = {
    'soc_0-5cm_mean': 'Soil Organic Carbon (0-5 cm)',
    'soc_5-15cm_mean': 'Soil Organic Carbon (5-15 cm)',
    'soc_15-30cm_mean': 'Soil Organic Carbon (15-30 cm)',
    'soc_30-60cm_mean': 'Soil Organic Carbon (30-60 cm)',
    'soc_60-100cm_mean': 'Soil Organic Carbon (60-100 cm)',
    'soc_100-200cm_mean': 'Soil Organic Carbon (100-200 cm)'
}
for band in ['soc_0-5cm_mean', 'soc_5-15cm_mean', 'soc_15-30cm_mean', 'soc_30-60cm_mean', 'soc_60-100cm_mean', 'soc_100-200cm_mean']:
    try:
        add_feature(thawdb, point_collection, carbon, ee.Reducer.mean(), 250, bandmap[band], band)
        print('Added', bandmap[band], 'from SoilGrids')
    except:
        print('Could not add', bandmap[band], 'from SoilGrids')

nitrogen = ee.Image('projects/soilgrids-isric/nitrogen_mean')
bandmap = {
    'nitrogen_0-5cm_mean': 'Nitrogen (0-5 cm)',
    'nitrogen_5-15cm_mean': 'Nitrogen (5-15 cm)',
    'nitrogen_15-30cm_mean': 'Nitrogen (15-30 cm)',
    'nitrogen_30-60cm_mean': 'Nitrogen (30-60 cm)',
    'nitrogen_60-100cm_mean': 'Nitrogen (60-100 cm)',
    'nitrogen_100-200cm_mean': 'Nitrogen (100-200 cm)'
}
for band in ['nitrogen_0-5cm_mean', 'nitrogen_5-15cm_mean', 'nitrogen_15-30cm_mean', 'nitrogen_30-60cm_mean', 'nitrogen_60-100cm_mean', 'nitrogen_100-200cm_mean']:
    try:
        add_feature(thawdb, point_collection, nitrogen, ee.Reducer.mean(), 250, bandmap[band], band)
        print('Added', bandmap[band], 'from SoilGrids')
    except:
        print('Could not add', bandmap[band], 'from SoilGrids')

clays = ee.Image('projects/soilgrids-isric/clay_mean')
bandmap = {
    'clay_0-5cm_mean': 'Clay (0-5 cm)',
    'clay_5-15cm_mean': 'Clay (5-15 cm)',
    'clay_15-30cm_mean': 'Clay (15-30 cm)',
    'clay_30-60cm_mean': 'Clay (30-60 cm)',
    'clay_60-100cm_mean': 'Clay (60-100 cm)',
    'clay_100-200cm_mean': 'Clay (100-200 cm)'
}
for band in ['clay_0-5cm_mean', 'clay_5-15cm_mean', 'clay_15-30cm_mean', 'clay_30-60cm_mean', 'clay_60-100cm_mean', 'clay_100-200cm_mean']:
    try:
        add_feature(thawdb, point_collection, clays, ee.Reducer.mean(), 250, bandmap[band], band)
        print('Added', bandmap[band], 'from SoilGrids')
    except:
        print('Could not add', bandmap[band], 'from SoilGrids')

sands = ee.Image('projects/soilgrids-isric/sand_mean')
bandmap = {
    'sand_0-5cm_mean': 'Sand (0-5 cm)',
    'sand_5-15cm_mean': 'Sand (5-15 cm)',
    'sand_15-30cm_mean': 'Sand (15-30 cm)',
    'sand_30-60cm_mean': 'Sand (30-60 cm)',
    'sand_60-100cm_mean': 'Sand (60-100 cm)',
    'sand_100-200cm_mean': 'Sand (100-200 cm)'
}
for band in ['sand_0-5cm_mean', 'sand_5-15cm_mean', 'sand_15-30cm_mean', 'sand_30-60cm_mean', 'sand_60-100cm_mean', 'sand_100-200cm_mean']:
    try:
        add_feature(thawdb, point_collection, sands, ee.Reducer.mean(), 250, bandmap[band], band)
        print('Added', bandmap[band], 'from SoilGrids')
    except:
        print('Could not add', bandmap[band], 'from SoilGrids')

silts = ee.Image('projects/soilgrids-isric/silt_mean')
bandmap = {
    'silt_0-5cm_mean': 'Silt (0-5 cm)',
    'silt_5-15cm_mean': 'Silt (5-15 cm)',
    'silt_15-30cm_mean': 'Silt (15-30 cm)',
    'silt_30-60cm_mean': 'Silt (30-60 cm)',
    'silt_60-100cm_mean': 'Silt (60-100 cm)',
    'silt_100-200cm_mean': 'Silt (100-200 cm)'
}
for band in ['silt_0-5cm_mean', 'silt_5-15cm_mean', 'silt_15-30cm_mean', 'silt_30-60cm_mean', 'silt_60-100cm_mean', 'silt_100-200cm_mean']:
    try:
        add_feature(thawdb, point_collection, silts, ee.Reducer.mean(), 250, bandmap[band], band)
        print('Added', bandmap[band], 'from SoilGrids')
    except:
        print('Could not add', bandmap[band], 'from SoilGrids')

density = ee.Image('projects/soilgrids-isric/bdod_mean')
bandmap = {
    'bdod_0-5cm_mean': 'Bulk Density (0-5 cm)',
    'bdod_5-15cm_mean': 'Bulk Density (5-15 cm)',
    'bdod_15-30cm_mean': 'Bulk Density (15-30 cm)',
    'bdod_30-60cm_mean': 'Bulk Density (30-60 cm)',
    'bdod_60-100cm_mean': 'Bulk Density (60-100 cm)',
    'bdod_100-200cm_mean': 'Bulk Density (100-200 cm)'
}
for band in ['bdod_0-5cm_mean', 'bdod_5-15cm_mean', 'bdod_15-30cm_mean', 'bdod_30-60cm_mean', 'bdod_60-100cm_mean', 'bdod_100-200cm_mean']:
    try:
        add_feature(thawdb, point_collection, density, ee.Reducer.mean(), 250, bandmap[band], band)
        print('Added', bandmap[band], 'from SoilGrids')
    except:
        print('Could not add', bandmap[band], 'from SoilGrids')

# GEE track: maximum fire temperature re-derived inline from FIRMS (no asset).
# T30: FIRMS max is a ~9,000-image temporal reduction; the old add_feature path
# hung >26 min at full N, so this feature uses the shared-computation reduceRegions
# fallback (ee_sampling), which computes the reduction once and reads all points
# from it (the same "compute once" principle the datacube path relies on).
# Sampled at 4 km: reduceRegions cost scales with tile count, and 1 km was killed
# >60 min at full N while 4 km completed in ~2.5 min. 4 km is also the grid the
# datacube serves FIRMS on (build_prediction_data.py), so training and inference
# treat the fire layer at the same resolution.
try:
    firms = gee_features.max_fire_temp()
    ee_sampling.add_feature_reduceregions(thawdb, lons, lats, firms, ee.Reducer.mean(), 4000, 'Maximum Fire Temperature', 'T21')
    print('Added maximum fire temperature from FIRMS')
except Exception as e:
    failed_features.append(('Maximum Fire Temperature', repr(e)))
    print('Could not add maximum fire temperature from FIRMS:', e)

# GEE track: SWE + SWE/precip/temp trends re-derived inline from Daymet V4.
# T30: same "deep temporal reduction" shape as FIRMS, but never reached in T30
# testing, so we have no evidence they fail. Kept on the old add_feature path
# pending a full-N probe; if a feature exceeds the 10-min cap, flip it with a
# one-line change:
#     add_feature(thawdb, point_collection, IMG, ee.Reducer.mean(), SCALE, NAME, BAND)
#   ->
#     ee_sampling.add_feature_reduceregions(thawdb, lons, lats, IMG, ee.Reducer.mean(), SCALE, NAME, BAND)
try:
    add_feature(thawdb, point_collection, gee_features.mean_annual_swe(), ee.Reducer.mean(), 1000, 'Mean Annual SWE', 'swe')
    print('Added mean annual SWE from Daymet V4')
except Exception as e:
    failed_features.append(('Mean Annual SWE', repr(e)))
    print('Could not add mean annual SWE from Daymet V4:', e)

try:
    add_feature(thawdb, point_collection, gee_features.swe_trend(), ee.Reducer.mean(), 1000, 'Trend in SWE', 'scale')
    print('Added trend in SWE, derived from Daymet V4')
except Exception as e:
    failed_features.append(('Trend in SWE', repr(e)))
    print('Could not add trend in SWE, derived from Daymet V4:', e)

try:
    add_feature(thawdb, point_collection, gee_features.precip_trend(), ee.Reducer.mean(), 1000, 'Trend in precipitation', 'scale')
    print('Added trend in precipitation, derived from Daymet V4')
except Exception as e:
    failed_features.append(('Trend in precipitation', repr(e)))
    print('Could not add trend in precipitation, derived from Daymet V4:', e)

try:
    add_feature(thawdb, point_collection, gee_features.temp_trend(), ee.Reducer.mean(), 1000, 'Trend in temperature', 'scale')
    print('Added trend in temperature, derived from Daymet V4')
except Exception as e:
    failed_features.append(('Trend in temperature', repr(e)))
    print('Could not add trend in temperature, derived from Daymet V4:', e)

# --------------------------------------------------------------------------
# LOCAL track: nearest-sample downloaded rasters at the point coordinates for
# the four features with no GEE-catalog upstream (local_rasters.py). Land Cover
# and Vegetation Mode stay raw integer codes here; clean_feature_table.py
# one-hot encodes them (0 / nodata -> 'NaN' bucket, dropped there).
# --------------------------------------------------------------------------
# (lons, lats defined once near the point collection above.)

# Land cover (NLCD 2016): missing -> code 0 so clean's land_cover_labels[0]='NaN'.
lc = local_rasters.sample_points(local_rasters.NLCD_IMG, lons, lats)
thawdb['Land Cover'] = np.where(np.isnan(lc), 0.0, lc)
print('Added NLCD land cover (LOCAL)')

# Vegetation mode (ALFRESCO): keep NaN for nodata; clean skips the NaN category.
thawdb['Vegetation Mode'] = local_rasters.sample_points(local_rasters.VEGMODE_TIF, lons, lats)
print('Added ALFRESCO vegetation mode (LOCAL)')

# Flammability index (ALFRESCO), continuous.
thawdb['Flammability Index'] = local_rasters.sample_points(local_rasters.FLAMMABILITY_TIF, lons, lats)
print('Added ALFRESCO flammability index (LOCAL)')

# --------------------------------------------------------------------------
# T30: end-of-run import report. Keep per-feature failures non-fatal (so a late
# failure never discards hours of unrelated feature work), but surface anything
# that raised or came through entirely empty, so an incomplete table can never
# pass silently downstream.
# --------------------------------------------------------------------------
all_nan = [c for c in thawdb.columns
           if np.issubdtype(thawdb[c].dtype, np.number) and thawdb[c].isna().all()]

print('\n' + '=' * 70)
print('[T30] Feature import report')
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

# Save the updated feature table
print(thawdb.columns)
print(thawdb.shape)
thawdb.to_csv(data / 'features_dirty.csv', index = False)
