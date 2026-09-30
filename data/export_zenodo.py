"""Write the Zenodo data record: CF-1.8 NetCDF-4 files, the training table, and a data dictionary.

    poetry run python data/export_zenodo.py

Reads data/susceptibility.nc, data/aoa.nc, data/prediction_data.nc, and
data/features_clean.csv. Writes to data/zenodo/ (git-ignored):

  susceptibility.nc       log_evidence, DI, inside_aoa on the 1 km lat/lon grid
  prediction_datacube.nc  the 70 model features, in the schema of data/prediction_data.nc
  features_clean.csv      the training feature table
  data_dictionary.csv     one row per features_clean.csv column
"""

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA

OUT = DATA / 'zenodo'

DATA_DOI = '10.5281/zenodo.23068691'
PAPER = ("Pierce, E., Overeem, I., Webb, H., Rozmiarek, K., & Turetsky, M. (in review). "
         "Susceptibility to Abrupt Thaw across Alaska's Permafrost Landscapes. Earth's Future.")
SOURCE = 'https://github.com/ethan-pierce/abrupt-thaw-indicators (tag v1.0.0)'
INSTITUTION = ('Thayer School of Engineering, Dartmouth College; Institute of Arctic and '
               'Alpine Research, University of Colorado Boulder')
LICENSE = 'CC-BY-4.0'

THAWDB = 'Alaska Permafrost Thaw Database v2.0.0'
DEM = 'USGS 3DEP 10 m DEM'
MERIT = 'MERIT Hydro v1.0.1'
WORLDCLIM = 'WorldClim v1.4 bioclimatic variables'
DAYMET = 'Daymet V4, 1991-2020'
MODIS = 'MODIS MCD64A1 v6.1, 2001-2024'
ALFRESCO_FLAM = 'ALFRESCO relative flammability, CRU TS4.0 historical 1900-1999'
ALFRESCO_VEG = 'ALFRESCO vegetation mode, historical 1950-2008'
IRYP = 'IRYP v2 (confirmed yedoma)'
NLCD = 'NLCD 2016 Alaska'
SOILGRIDS = 'SoilGrids 2.0 (depth-weighted mean)'

# name -> (units, source product, type)
COLUMNS = {
    'Latitude': ('degrees_north', THAWDB, 'coordinate'),
    'Longitude': ('degrees_east', THAWDB, 'coordinate'),
    'Class': ('1', THAWDB, 'label (0 = Abrupt, 1 = Non-abrupt)'),
    'Thaw Type': ('', THAWDB, 'label'),
    'Thaw Database Row': ('', THAWDB, 'identifier (0-based row in the Thaw Database CSV)'),
    'Elevation': ('m', DEM, 'continuous'),
    'Slope': ('degree', DEM, 'continuous'),
    'Northness': ('1', DEM, 'continuous (cosine of aspect)'),
    'Eastness': ('1', DEM, 'continuous (sine of aspect)'),
    'Mean curvature (500 m)': ('m-1', DEM, 'continuous'),
    'Mean curvature (2 km)': ('m-1', DEM, 'continuous'),
    'Height Above Nearest Drainage': ('m', MERIT, 'continuous'),
    'Upstream Area': ('km2', MERIT, 'continuous'),
    'Annual Mean Temperature': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Mean Diurnal Range': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Isothermality': ('percent', WORLDCLIM, 'continuous'),
    'Temperature Seasonality': ('0.001 degC', WORLDCLIM, 'continuous (standard deviation)'),
    'Max Temperature of Warmest Month': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Min Temperature of Coldest Month': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Temperature Annual Range': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Mean Temperature of Wettest Quarter': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Mean Temperature of Driest Quarter': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Mean Temperature of Warmest Quarter': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Mean Temperature of Coldest Quarter': ('0.1 degC', WORLDCLIM, 'continuous'),
    'Annual Precipitation': ('mm', WORLDCLIM, 'continuous'),
    'Precipitation of Wettest Month': ('mm', WORLDCLIM, 'continuous'),
    'Precipitation of Driest Month': ('mm', WORLDCLIM, 'continuous'),
    'Precipitation Seasonality': ('percent', WORLDCLIM, 'continuous (coefficient of variation)'),
    'Precipitation of Wettest Quarter': ('mm', WORLDCLIM, 'continuous'),
    'Precipitation of Driest Quarter': ('mm', WORLDCLIM, 'continuous'),
    'Precipitation of Warmest Quarter': ('mm', WORLDCLIM, 'continuous'),
    'Precipitation of Coldest Quarter': ('mm', WORLDCLIM, 'continuous'),
    'Mean Annual SWE': ('kg m-2', DAYMET, 'continuous'),
    'Trend in SWE': ('kg m-2 yr-1', DAYMET, 'continuous (OLS slope of annual mean)'),
    'Trend in precipitation': ('mm yr-1', DAYMET, 'continuous (OLS slope of annual total)'),
    'Trend in temperature': ('degC yr-1', DAYMET, 'continuous (OLS slope of annual mean daily maximum)'),
    'Time Since Last Fire': ('yr', MODIS, 'continuous (years before 2025; 24 = no burn detected)'),
    'Burn Count': ('1', MODIS, 'count (months with a detected burn)'),
    'Flammability Index': ('1', ALFRESCO_FLAM, 'continuous'),
    'Yedoma': ('1', IRYP, 'binary'),
    **{f'Soil Organic Carbon ({d})': ('dg kg-1', SOILGRIDS, 'continuous') for d in ('0-30 cm', '30-200 cm')},
    **{f'Nitrogen ({d})': ('cg kg-1', SOILGRIDS, 'continuous') for d in ('0-30 cm', '30-200 cm')},
    **{f'Bulk Density ({d})': ('cg cm-3', SOILGRIDS, 'continuous') for d in ('0-30 cm', '30-200 cm')},
    **{f'Sand ({d})': ('g kg-1', SOILGRIDS, 'continuous') for d in ('0-30 cm', '30-200 cm')},
    **{f'Clay ({d})': ('g kg-1', SOILGRIDS, 'continuous') for d in ('0-30 cm', '30-200 cm')},
}


def column_meta(name):
    if name.startswith('Land Cover ('):
        return ('1', NLCD, 'binary (one-hot)')
    if name.startswith('Vegetation Mode ('):
        return ('1', ALFRESCO_VEG, 'binary (one-hot)')
    return COLUMNS[name]


def global_attrs(title, summary):
    return {
        'title': title,
        'summary': summary,
        'institution': INSTITUTION,
        'source': SOURCE,
        'references': f'https://doi.org/{DATA_DOI}; {PAPER}',
        'license': LICENSE,
        'Conventions': 'CF-1.8',
        'history': f'{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} written by data/export_zenodo.py',
    }


def lat_lon(ds):
    """Collapse the 2-D longitude/latitude (fill -9999 off the ROI) to 1-D axes."""
    lon2d, lat2d = ds['longitude'].values, ds['latitude'].values
    ok = (np.abs(lon2d) <= 180) & (np.abs(lat2d) <= 90)
    lon2d, lat2d = np.where(ok, lon2d, np.nan), np.where(ok, lat2d, np.nan)
    lon, lat = np.nanmedian(lon2d, axis=0), np.nanmedian(lat2d, axis=1)
    assert np.array_equal(np.broadcast_to(lon, lon2d.shape)[ok], lon2d[ok]), 'grid not separable in lon'
    assert np.array_equal(np.broadcast_to(lat[:, None], lat2d.shape)[ok], lat2d[ok]), 'grid not separable in lat'
    lon_da = xr.DataArray(lon, dims='lon', attrs={
        'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'})
    lat_da = xr.DataArray(lat, dims='lat', attrs={
        'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'})
    return lat_da, lon_da


def crs_var():
    return xr.DataArray(np.int32(0), attrs={
        'grid_mapping_name': 'latitude_longitude',
        'longitude_of_prime_meridian': 0.0,
        'semi_major_axis': 6378137.0,
        'inverse_flattening': 298.257223563,
        'crs_wkt': CRS.from_epsg(4326).to_wkt(),
    })


def float_encoding(chunks):
    return {'dtype': 'float32', '_FillValue': np.float32(np.nan), 'zlib': True,
            'complevel': 4, 'shuffle': True, 'chunksizes': chunks}


def export_susceptibility(lat, lon):
    sus = xr.open_dataset(DATA / 'susceptibility.nc')
    aoa = xr.open_dataset(DATA / 'aoa.nc')
    log_evidence = sus['log_evidence'].values
    di = aoa['DI'].values
    inside = aoa['inside_aoa'].values
    assert np.array_equal(np.isnan(log_evidence), np.isnan(di)), 'log_evidence and DI masks differ'
    inside_aoa = np.where(np.isnan(inside), -1, inside).astype(np.int8)

    dims = ('lat', 'lon')
    ds = xr.Dataset(
        {
            'log_evidence': (dims, log_evidence, {
                'long_name': 'abrupt-thaw susceptibility (log-evidence)',
                'units': '1',
                'description': ('Log-likelihood ratio of abrupt vs. non-abrupt thaw given local '
                                'features: logit(P_model(abrupt|x)) - logit(sample abrupt fraction). '
                                '0 is neutral; positive values favor abrupt thaw. Not a probability.'),
                'grid_mapping': 'crs'}),
            'DI': (dims, di, {
                'long_name': 'dissimilarity index',
                'units': '1',
                'description': ('Importance-weighted distance in feature space to the nearest training '
                                'point, divided by the mean pairwise training distance.'),
                'grid_mapping': 'crs'}),
            'inside_aoa': (dims, inside_aoa, {
                'long_name': 'inside area of applicability',
                'flag_values': np.array([0, 1], dtype=np.int8),
                'flag_meanings': 'outside_aoa inside_aoa',
                'description': 'Cells with DI at or below the area-of-applicability threshold.',
                'aoa_threshold': float(aoa.attrs['aoa_threshold']),
                'grid_mapping': 'crs'}),
            'crs': crs_var(),
        },
        coords={'lat': lat, 'lon': lon},
        attrs=global_attrs(
            'Abrupt-thaw susceptibility across Alaska\'s permafrost domain',
            ('Log-evidence index of abrupt vs. non-abrupt permafrost thaw, dissimilarity index, and '
             'area-of-applicability flag on a 1 km (0.00898 degree) grid. Cells outside the '
             'permafrost domain (Obu et al. 2019 permafrost probability of 0) are missing.')),
    )
    chunks = (512, 512)
    encoding = {
        'log_evidence': float_encoding(chunks),
        'DI': float_encoding(chunks),
        'inside_aoa': {'dtype': 'int8', '_FillValue': np.int8(-1), 'zlib': True,
                       'complevel': 4, 'shuffle': True, 'chunksizes': chunks},
        'lat': {'_FillValue': None}, 'lon': {'_FillValue': None},
    }
    path = OUT / 'susceptibility.nc'
    ds.to_netcdf(path, format='NETCDF4', encoding=encoding)
    return path


def export_datacube():
    """Same schema as data/prediction_data.nc, which the pipeline reads, plus metadata."""
    src = xr.open_dataset(DATA / 'prediction_data.nc')
    stack = np.empty(src['feature_stack'].shape, dtype=np.float32)
    for y0 in range(0, stack.shape[0], 128):
        stack[y0:y0 + 128] = src['feature_stack'][y0:y0 + 128].values

    meta = [column_meta(str(n)) for n in src['feature'].values]
    ds = src.drop_vars('feature_stack')
    ds['feature_stack'] = (('y', 'x', 'feature'), stack, {
        'long_name': 'model input features',
        'description': ('The 70 geospatial features the classifier scores, sampled at each cell '
                        'centre. Units per feature are in feature_units.'),
        'grid_mapping': 'crs'})
    ds['feature_units'] = ('feature', np.array([m[0] for m in meta], dtype=object),
                           {'long_name': 'units of each feature'})
    ds['feature_source'] = ('feature', np.array([m[1] for m in meta], dtype=object),
                            {'long_name': 'source data product of each feature'})
    ds['crs'] = crs_var()
    for name, axis in (('longitude', 'degrees_east'), ('latitude', 'degrees_north')):
        ds[name].attrs.update({'standard_name': name, 'units': axis,
                               'description': 'Cell-centre coordinate; -9999 outside the region of interest.'})
    ds['x'].attrs['long_name'] = 'column index (west to east)'
    ds['y'].attrs['long_name'] = 'row index (north to south)'
    ds.attrs.update(global_attrs(
        'Prediction datacube of geospatial thaw indicators across Alaska',
        ('The 70 geospatial features used to predict thaw mode, on a 1 km (0.00898 degree) '
         'EPSG:4326 grid. Same schema as data/prediction_data.nc in the code repository.')))

    encoding = {'feature_stack': float_encoding((128, 128, 70)),
                'longitude': {'_FillValue': None}, 'latitude': {'_FillValue': None}}
    path = OUT / 'prediction_datacube.nc'
    ds.to_netcdf(path, format='NETCDF4', encoding=encoding)
    return path


def export_tables():
    table = OUT / 'features_clean.csv'
    shutil.copyfile(DATA / 'features_clean.csv', table)
    columns = pd.read_csv(table, nrows=0).columns
    rows = [(c, *column_meta(c)) for c in columns]
    dictionary = OUT / 'data_dictionary.csv'
    pd.DataFrame(rows, columns=['name', 'units', 'source', 'type']).to_csv(dictionary, index=False)
    return table, dictionary


def main():
    OUT.mkdir(exist_ok=True)
    lat, lon = lat_lon(xr.open_dataset(DATA / 'susceptibility.nc'))
    for other in ('aoa.nc', 'prediction_data.nc'):
        o_lat, o_lon = lat_lon(xr.open_dataset(DATA / other))
        assert np.array_equal(o_lat, lat) and np.array_equal(o_lon, lon), f'{other} grid differs'
    for path in (export_susceptibility(lat, lon), export_datacube(), *export_tables()):
        print(f'wrote {path} ({path.stat().st_size / 2**20:,.1f} MiB)')


if __name__ == '__main__':
    main()
