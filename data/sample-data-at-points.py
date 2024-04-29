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
worldclim = ee.ImageCollection('WORLDCLIM/V1/MONTHLY')
varmap = {
    'tavg': 'mean temperature',
    'tmin': 'minimum temperature',
    'tmax': 'maximum temperature',
    'prec': 'precipitation'
}
monthmap = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
# for var in ['tavg', 'tmin', 'tmax', 'prec']:
for var in ['prec']:
    print('Adding', varmap[var], 'to features...')
    for month in range(1, 13):
        if var != 'prec':
            clim = worldclim.select(var).filter(ee.Filter.eq('month', month)).first().multiply(0.1)
        else:
            clim = worldclim.select(var).filter(ee.Filter.eq('month', month)).first()

        add_feature(clim, ee.Reducer.mean(), 1000, f'{monthmap[month]} {varmap[var]}', var)

        print('Added', monthmap[month], varmap[var])
        break

print(feats.head)
# Save the updated feature table
feats.to_csv(os.path.join(DATA, 'feature-table.csv'), index = False)
