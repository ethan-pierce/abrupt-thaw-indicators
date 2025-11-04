"""Build a feature table for the thaw database."""

import ee
ee.Authenticate()
ee.Initialize(project = 'ee-abrupt-thaw')


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = Path(__file__).parent

thawdb = pd.read_csv(data / 'Alaska_Permafrost_Thaw_Database_v1.0.0-alpha.csv', sep = ',', encoding = 'latin1')

print(thawdb['ThawType'].value_counts()) # 7.28% gradual, 92.72% abrupt
thawdb['Class'] = np.where(thawdb['ThawType'] == 'Abrupt', 1, 0) # ABRUPT = 1, GRADUAL = 0

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

# VARIABLE: Land cover
try:
    landcover = ee.Image('projects/ee-abrupt-thaw/assets/NLCD-2016')
    add_feature(thawdb, point_collection, landcover, ee.Reducer.mode(), 30, 'Land Cover', 'b1')
    print('Added NLCD land cover')
except:
    print('Could not add NLCD land cover')

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

try:
    curve500 = ee.Image('projects/ee-abrupt-thaw/assets/AK-curvature-500m')
    add_feature(thawdb, point_collection, curve500, ee.Reducer.mean(), 100, 'Mean curvature (500 m)', 'MeanCurvature')
    print('Added mean 500m curvature derived from TAGEE algorithm applied to USGS 3DEP elevation')
except:
    print('Could not add mean 500m curvature derived from TAGEE algorithm applied to USGS 3DEP elevation')

try:
    curve2k = ee.Image('projects/ee-abrupt-thaw/assets/AK-curvature-2k')
    add_feature(thawdb, point_collection, curve2k, ee.Reducer.mean(), 100, 'Mean curvature (2 km)', 'MeanCurvature')
    print('Added mean 2km curvature derived from TAGEE algorithm applied to USGS 3DEP elevation')
except:
    print('Could not add mean 2km curvature derived from TAGEE algorithm applied to USGS 3DEP elevation')

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

# VARIABLES: flammability index
try:
    flammability = ee.Image('projects/ee-abrupt-thaw/assets/ALFRESCO-historical-flammability')
    add_feature(thawdb, point_collection, flammability, ee.Reducer.mean(), 1000, 'Flammability Index', 'b1')
    print('Added ALFRESCO flammability index')
except:
    print('Could not add ALFRESCO flammability index')

# VARIABLES: vegetation mode
try:
    vegetation = ee.Image('projects/ee-abrupt-thaw/assets/ALFRESCO-historical-vegetation-mode')
    add_feature(thawdb, point_collection, vegetation, ee.Reducer.mode(), 1000, 'Vegetation Mode', 'b1')
    print('Added ALFRESCO vegetation mode')
except:
    print('Could not add ALFRESCO vegetation mode')

# VARIABLES: maximum fire temperature
try:
    firms = ee.Image('projects/ee-abrupt-thaw/assets/max-fire-temp')
    add_feature(thawdb, point_collection, firms, ee.Reducer.mean(), 1000, 'Maximum Fire Temperature', 'T21')
    print('Added maximum fire temperature from FIRMS')
except:
    print('Could not add maximum fire temperature from FIRMS')

# VARIABLES: swe, change in swe, change in tmax, change in prcp
try:
    swe = ee.Image('projects/ee-abrupt-thaw/assets/ee-mean-annual-swe')
    add_feature(thawdb, point_collection, swe, ee.Reducer.mean(), 1000, 'Mean Annual SWE', 'swe')
    print('Added mean annual SWE from Daymet V4')
except:
    print('Could not add mean annual SWE from Daymet V4')

try:
    swe_trend = ee.Image('projects/ee-abrupt-thaw/assets/annual-swe-trend')
    add_feature(thawdb, point_collection, swe_trend.select('scale'), ee.Reducer.mean(), 1000, 'Trend in SWE', 'scale')
    print('Added trend in SWE, derived from Daymet V4')
except:
    print('Could not add trend in SWE, derived from Daymet V4')

try:
    precip_trend = ee.Image('projects/ee-abrupt-thaw/assets/annual-precip-trend')
    add_feature(thawdb, point_collection, precip_trend.select('scale'), ee.Reducer.mean(), 1000, 'Trend in precipitation', 'scale')
    print('Added trend in precipitation, derived from Daymet V4')
except:
    print('Could not add trend in precipitation, derived from Daymet V4')

try:
    tmax_trend = ee.Image('projects/ee-abrupt-thaw/assets/temp-trend')
    add_feature(thawdb, point_collection, tmax_trend.select('scale'), ee.Reducer.mean(), 1000, 'Trend in temperature', 'scale')
    print('Added trend in temperature, derived from Daymet V4')
except:
    print('Could not add trend in temperature, derived from Daymet V4')

# VARIABLES: projected summer and winter temp change, precip change
try:
    projected_summer_temp = ee.Image('projects/ee-abrupt-thaw/assets/summer-temperature-trend')
    add_feature(thawdb, point_collection, projected_summer_temp, ee.Reducer.mean(), 1000, 'Projected summer temperature change', 'b1')
    print('Added projected summer temperature change from CRU TS3.1')
except:
    print('Could not add projected summer temperature change from CRU TS3.1')

try:
    projected_winter_temp = ee.Image('projects/ee-abrupt-thaw/assets/winter-temperature-trend')
    add_feature(thawdb, point_collection, projected_winter_temp, ee.Reducer.mean(), 1000, 'Projected winter temperature change', 'b1')
    print('Added projected winter temperature change from CRU TS3.1')
except:
    print('Could not add projected winter temperature change from CRU TS3.1')

try:
    projected_precip = ee.Image('projects/ee-abrupt-thaw/assets/annual-precipitation-trend')
    add_feature(thawdb, point_collection, projected_precip, ee.Reducer.mean(), 1000, 'Projected precipitation change', 'b1')
    print('Added projected precipitation change from CRU TS3.1')
except:
    print('Could not add projected precipitation change from CRU TS3.1')

# Save the updated feature table
print(thawdb.columns)
print(thawdb.shape)
thawdb.to_csv(data / 'features_dirty.csv', index = False)
