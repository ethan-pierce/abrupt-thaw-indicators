"""Sample data at site locations using GEE tools."""

import ee
ee.Authenticate()
ee.Initialize(project = 'ee-abrupt-thaw')

from settings import DATA

import os
import numpy as np
import pandas as pd
import geemap
import matplotlib.pyplot as plt

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
    df = ee.data.computeFeatures({
        'expression': values,
        'fileFormat': 'PANDAS_DATAFRAME'
    })
    feats[name] = [row[band] for idx, row in df.iterrows()]

# Load the feature table
feats = pd.read_csv(os.path.join(DATA, 'feature-table.csv'))
coords = [ee.Feature(ee.Geometry.Point([lon, lat])) for lon, lat in zip(feats['Longitude'], feats['Latitude'])]
points = ee.FeatureCollection(coords)

# VARIABLE: Land cover
# landcover = ee.Image('projects/ee-abrupt-thaw/assets/NLCD-2016')
# add_feature(landcover, ee.Reducer.mode(), 30, 'Land Cover', 'b1')

# VARIABLES: Elevation and Slope
# elevation = ee.Image('USGS/3DEP/10m').select('elevation')
# add_feature(elevation, ee.Reducer.mean(), 10, 'Elevation', 'elevation')
# slope = ee.Terrain.slope(elevation)
# add_feature(slope, ee.Reducer.mean(), 10, 'Slope', 'slope')

# VARIABLES: min/max/mean temperature and net precipitation climatologies (1960 - 1990)
# worldclim = ee.ImageCollection('WORLDCLIM/V1/MONTHLY')
# varmap = {
#     'tavg': 'mean temperature',
#     'tmin': 'minimum temperature',
#     'tmax': 'maximum temperature',
#     'prec': 'precipitation'
# }
# monthmap = {
#     1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
#     7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
# }
# for var in ['tavg', 'prec']:
#     print('Adding', varmap[var], 'to features...')
#     for month in range(1, 13):
#         if var != 'prec':
#             clim = worldclim.select(var).filter(ee.Filter.eq('month', month)).first().multiply(0.1)
#         else:
#             clim = worldclim.select(var).filter(ee.Filter.eq('month', month)).first()

#         add_feature(clim, ee.Reducer.mean(), 1000, f'{monthmap[month]} {varmap[var]}', var)

#         print('Added', monthmap[month], varmap[var])
# VARIABLES: temperature amplitude
# tmax = worldclim.select('tmax').max().multiply(0.1)
# tmin = worldclim.select('tmin').min().multiply(0.1)
# add_feature(tmax.subtract(tmin), ee.Reducer.mean(), 1000, 'Temperature amplitude', 'tmax')

# VARIABLES: bioclimatic variables
# bioclim = ee.Image('WORLDCLIM/V1/BIO')
# biovars = {
#     'bio01': 'Annual Mean Temperature',
#     'bio02': 'Mean Diurnal Range',
#     'bio03': 'Isothermality',
#     'bio04': 'Temperature Seasonality',
#     'bio05': 'Max Temperature of Warmest Month',
#     'bio06': 'Min Temperature of Coldest Month',
#     'bio07': 'Temperature Annual Range',
#     'bio08': 'Mean Temperature of Wettest Quarter',
#     'bio09': 'Mean Temperature of Driest Quarter',
#     'bio10': 'Mean Temperature of Warmest Quarter',
#     'bio11': 'Mean Temperature of Coldest Quarter',
#     'bio12': 'Annual Precipitation',
#     'bio13': 'Precipitation of Wettest Month',
#     'bio14': 'Precipitation of Driest Month',
#     'bio15': 'Precipitation Seasonality',
#     'bio16': 'Precipitation of Wettest Quarter',
#     'bio17': 'Precipitation of Driest Quarter',
#     'bio18': 'Precipitation of Warmest Quarter',
#     'bio19': 'Precipitation of Coldest Quarter'
# }
# for band, name in biovars.items():
#     add_feature(bioclim, ee.Reducer.mean(), 1000, name, band)
#     print('Added', name)

# VARIABLES: soil texture, nitrogen, organic carbon
# carbon = ee.Image('projects/soilgrids-isric/soc_mean')
# bandmap = {
#     'soc_0-5cm_mean': 'Soil Organic Carbon (0-5 cm)',
#     'soc_5-15cm_mean': 'Soil Organic Carbon (5-15 cm)',
#     'soc_15-30cm_mean': 'Soil Organic Carbon (15-30 cm)',
#     'soc_30-60cm_mean': 'Soil Organic Carbon (30-60 cm)',
#     'soc_60-100cm_mean': 'Soil Organic Carbon (60-100 cm)',
#     'soc_100-200cm_mean': 'Soil Organic Carbon (100-200 cm)'
# }
# for band in ['soc_0-5cm_mean', 'soc_5-15cm_mean', 'soc_15-30cm_mean', 'soc_30-60cm_mean', 'soc_60-100cm_mean', 'soc_100-200cm_mean']:
#     add_feature(carbon, ee.Reducer.mean(), 250, bandmap[band], band)
#     print('Added', bandmap[band])

# nitrogen = ee.Image('projects/soilgrids-isric/nitrogen_mean')
# bandmap = {
#     'nitrogen_0-5cm_mean': 'Nitrogen (0-5 cm)',
#     'nitrogen_5-15cm_mean': 'Nitrogen (5-15 cm)',
#     'nitrogen_15-30cm_mean': 'Nitrogen (15-30 cm)',
#     'nitrogen_30-60cm_mean': 'Nitrogen (30-60 cm)',
#     'nitrogen_60-100cm_mean': 'Nitrogen (60-100 cm)',
#     'nitrogen_100-200cm_mean': 'Nitrogen (100-200 cm)'
# }
# for band in ['nitrogen_0-5cm_mean', 'nitrogen_5-15cm_mean', 'nitrogen_15-30cm_mean', 'nitrogen_30-60cm_mean', 'nitrogen_60-100cm_mean', 'nitrogen_100-200cm_mean']:
#     add_feature(nitrogen, ee.Reducer.mean(), 250, bandmap[band], band)
#     print('Added', band)

# clays = ee.Image('projects/soilgrids-isric/clay_mean')
# bandmap = {
#     'clay_0-5cm_mean': 'Clay (0-5 cm)',
#     'clay_5-15cm_mean': 'Clay (5-15 cm)',
#     'clay_15-30cm_mean': 'Clay (15-30 cm)',
#     'clay_30-60cm_mean': 'Clay (30-60 cm)',
#     'clay_60-100cm_mean': 'Clay (60-100 cm)',
#     'clay_100-200cm_mean': 'Clay (100-200 cm)'
# }
# for band in ['clay_0-5cm_mean', 'clay_5-15cm_mean', 'clay_15-30cm_mean', 'clay_30-60cm_mean', 'clay_60-100cm_mean', 'clay_100-200cm_mean']:
#     add_feature(clays, ee.Reducer.mean(), 250, bandmap[band], band)
#     print('Added', band)

# sands = ee.Image('projects/soilgrids-isric/sand_mean')
# bandmap = {
#     'sand_0-5cm_mean': 'Sand (0-5 cm)',
#     'sand_5-15cm_mean': 'Sand (5-15 cm)',
#     'sand_15-30cm_mean': 'Sand (15-30 cm)',
#     'sand_30-60cm_mean': 'Sand (30-60 cm)',
#     'sand_60-100cm_mean': 'Sand (60-100 cm)',
#     'sand_100-200cm_mean': 'Sand (100-200 cm)'
# }
# for band in ['sand_0-5cm_mean', 'sand_5-15cm_mean', 'sand_15-30cm_mean', 'sand_30-60cm_mean', 'sand_60-100cm_mean', 'sand_100-200cm_mean']:
#     add_feature(sands, ee.Reducer.mean(), 250, bandmap[band], band)
#     print('Added', band)

# silts = ee.Image('projects/soilgrids-isric/silt_mean')
# bandmap = {
#     'silt_0-5cm_mean': 'Silt (0-5 cm)',
#     'silt_5-15cm_mean': 'Silt (5-15 cm)',
#     'silt_15-30cm_mean': 'Silt (15-30 cm)',
#     'silt_30-60cm_mean': 'Silt (30-60 cm)',
#     'silt_60-100cm_mean': 'Silt (60-100 cm)',
#     'silt_100-200cm_mean': 'Silt (100-200 cm)'
# }
# for band in ['silt_0-5cm_mean', 'silt_5-15cm_mean', 'silt_15-30cm_mean', 'silt_30-60cm_mean', 'silt_60-100cm_mean', 'silt_100-200cm_mean']:
#     add_feature(silts, ee.Reducer.mean(), 250, bandmap[band], band)
#     print('Added', band)

# density = ee.Image('projects/soilgrids-isric/bdod_mean')
# bandmap = {
#     'bdod_0-5cm_mean': 'Bulk Density (0-5 cm)',
#     'bdod_5-15cm_mean': 'Bulk Density (5-15 cm)',
#     'bdod_15-30cm_mean': 'Bulk Density (15-30 cm)',
#     'bdod_30-60cm_mean': 'Bulk Density (30-60 cm)',
#     'bdod_60-100cm_mean': 'Bulk Density (60-100 cm)',
#     'bdod_100-200cm_mean': 'Bulk Density (100-200 cm)'
# }
# for band in ['bdod_0-5cm_mean', 'bdod_5-15cm_mean', 'bdod_15-30cm_mean', 'bdod_30-60cm_mean', 'bdod_60-100cm_mean', 'bdod_100-200cm_mean']:
#     add_feature(density, ee.Reducer.mean(), 250, bandmap[band], band)
#     print('Added', band)

# VARIABLES: flammability index
# flammability = ee.Image('projects/ee-abrupt-thaw/assets/ALFRESCO-historical-flammability')
# add_feature(flammability, ee.Reducer.mean(), 1000, 'Flammability Index', 'b1')

# # VARIABLES: vegetation mode
# vegetation = ee.Image('projects/ee-abrupt-thaw/assets/ALFRESCO-historical-vegetation-mode')
# add_feature(vegetation, ee.Reducer.mode(), 1000, 'Vegetation Mode', 'b1')

# VARIABLES: maximum fire temperature
# firms = ee.Image('projects/ee-abrupt-thaw/assets/max-fire-temp')
# add_feature(firms, ee.Reducer.mean(), 1000, 'Maximum Fire Temperature', 'T21')

# VARIABLES: swe, change in swe, change in tmax, change in prcp
# swe = ee.Image('projects/ee-abrupt-thaw/assets/ee-mean-annual-swe')
# add_feature(swe, ee.Reducer.mean(), 1000, 'Mean Annual SWE', 'swe')

# swe_trend = ee.Image('projects/ee-abrupt-thaw/assets/annual-swe-trend')
# add_feature(swe_trend.select('scale'), ee.Reducer.mean(), 1000, 'Trend in SWE', 'scale')

# precip_trend = ee.Image('projects/ee-abrupt-thaw/assets/annual-precip-trend')
# add_feature(precip_trend.select('scale'), ee.Reducer.mean(), 1000, 'Trend in precipitation', 'scale')

# tmax_trend = ee.Image('projects/ee-abrupt-thaw/assets/temp-trend')
# add_feature(tmax_trend.select('scale'), ee.Reducer.mean(), 1000, 'Trend in temperature', 'scale')

# Save the updated feature table
print(feats.head)
print(feats.columns)
feats.to_csv(os.path.join(DATA, 'feature-table.csv'), index = False)
