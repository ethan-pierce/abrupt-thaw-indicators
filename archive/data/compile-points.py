"""Compile points from the Thaw Database and control point lists into one table."""

import ee
ee.Authenticate()
ee.Initialize(project = 'ee-abrupt-thaw')

import os
import numpy as np
import pandas as pd
from settings import DATA

thawdb = pd.read_csv(os.path.join(DATA, 'ThawDatabase06222023.csv'))
thawdb.drop(
    labels = thawdb.loc[
        (thawdb['Thaw Type (abrupt vs gradual)'] != 'Abrupt')
        & (thawdb['Thaw Type (abrupt vs gradual)'] != 'Gradual')
    ].index,
    axis = 0,
    inplace = True
)
thawdb['Type'] = np.where(thawdb['Thaw Type (abrupt vs gradual)'] == 'Abrupt', 1, 0)
thawdb.drop(
    labels = ['Authors', 'DOI', 'Feature Type', 'Site Name', 'Thaw Type (abrupt vs gradual)'],
    axis = 1,
    inplace = True
)

controls = pd.read_csv(os.path.join(DATA, 'controls.csv'))
controls['Latitude'] = controls['Y']
controls['Longitude'] = controls['X']
controls['Type'] = 0
controls.drop(
    labels = [
        'X', 'Y', 'Name', 'AltMode', 'Base', 'Field_1', 'FolderPath', 
        'HasLabel', 'LabelID', 'Name2', 'OID', 'PopupInfo', 'SymbolID'
    ],
    axis = 1,
    inplace = True
)

thawdb = pd.merge(thawdb, controls, how = 'outer')
thawdb.to_csv(os.path.join(DATA, 'verified-points.csv'), index = False)

synthetic = ee.data.listFeatures({
    'assetId': 'projects/ee-abrupt-thaw/assets/synthetic-examples',
    'fileFormat': 'PANDAS_DATAFRAME'
})
latitude = [synthetic['geo'][i]['coordinates'][1] for i in range(len(synthetic['geo']))]
longitude = [synthetic['geo'][i]['coordinates'][0] for i in range(len(synthetic['geo']))]
synthetic['Latitude'] = latitude
synthetic['Longitude'] = longitude
synthetic['Type'] = 0
synthetic.drop(labels = ['geo', 'b1'], axis = 1, inplace = True)

synthetic.to_csv(os.path.join(DATA, 'synthetic-points.csv'), index = False)

thawdb = pd.merge(thawdb, synthetic, how = 'outer')
print(thawdb.shape)
print(thawdb['Type'].value_counts())

thawdb.to_csv(os.path.join(DATA, 'feature-points.csv'), index = False)