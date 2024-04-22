"""Compile points from the Thaw Database and control point lists into one table."""

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
