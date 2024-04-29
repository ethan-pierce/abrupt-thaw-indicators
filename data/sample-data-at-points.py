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
    scale: float, 
    name: str,
    band: str,
    proj: str = 'EPSG:4326'
) -> float:
    """Sample a raster at the point corresponding to a single feature."""
    point = feat.geometry(proj = proj)
    value = image.sample(region = point, scale = scale, projection = proj)
    return feat.set({name: value.first().get(band)})

# Load the feature table
feats = pd.read_csv(os.path.join(DATA, 'feature-points.csv'))
coords = [ee.Feature(ee.Geometry.Point([lon, lat])) for lon, lat in zip(feats['Longitude'], feats['Latitude'])]
points = ee.FeatureCollection(coords)

# VARIABLE: Land cover
# landcover = ee.Image('projects/ee-abrupt-thaw/assets/NLCD-2016')
# landcover_vals = points.map(lambda feat: sample_raster(landcover, feat, 30, 'Land Cover', landcover.bandNames().get(0)))
# landcover_df = ee.data.computeFeatures({
#     'expression': landcover_vals,
#     'fileFormat': 'PANDAS_DATAFRAME'
# })
# feats['Land cover class'] = landcover_df['Land Cover']
# feats.to_csv(os.path.join(DATA, 'feature-table.csv'), index = False)


