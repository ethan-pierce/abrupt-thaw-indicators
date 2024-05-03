"""Clean the feature table and save it to a new file."""

import os
from settings import DATA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

feats = pd.read_csv(os.path.join(DATA, 'feature-table.csv'))
clean = feats.copy()

clean['Class'] = clean['Type']
clean = clean.drop('Type', axis = 1)

label = ['Class']
fillna = ['Maximum Fire Temperature']
categorical = ['Land Cover', 'Vegetation Mode']
land_cover_labels = {
    0: 'NaN',
    11: 'Open Water',
    12: 'Perennial Ice/Snow',
    21: 'Developed, Open Space',
    22: 'Developed, Low Intensity',
    23: 'Developed, Medium Intensity',
    24: 'Developed, High Intensity',
    31: 'Barren Land (Rock/Sand/Clay)',
    41: 'Deciduous Forest',
    42: 'Evergreen Forest',
    43: 'Mixed Forest',
    51: 'Dwarf Scrub',
    52: 'Shrub/Scrub',
    71: 'Grassland/Herbaceous',
    72: 'Sedge/Herbaceous',
    73: 'Lichens',
    74: 'Moss',
    81: 'Pasture/Hay',
    82: 'Cultivated Crops',
    90: 'Woody Wetlands',
    95: 'Emergent Herbaceous Wetlands'
}

vegetation_mode_labels = {
    0: 'NaN',
    1: 'Black spruce',
    2: 'White spruce',
    3: 'Deciduous forest',
    4: 'Shrub tundra',
    5: 'Graminoid tundra',
    6: 'Wetland tundra',
    7: 'Barren lichen moss',
    8: 'Temperate rainforest'
}

for col in clean.columns:
    if col in label:
        continue
    if col in fillna:
        clean[col] = np.where(np.isnan(clean[col]), 0.0, clean[col])
    if col in categorical:
        categories = clean[col].unique()
        for cat in categories:
            if col == 'Land Cover':
                clean[col + ' (' + land_cover_labels[cat] + ')'] = np.where(clean[col] == cat, 1, 0)
            if col == 'Vegetation Mode':
                if np.isnan(cat):
                    continue
                clean[col + ' (' + vegetation_mode_labels[cat] + ')'] = np.where(clean[col] == cat, 1, 0)
        clean = clean.drop(col, axis=1)
    else:
        continue
    
clean = clean.dropna(axis = 0, how = 'any')

clean.to_csv(os.path.join(DATA, 'clean-feature-table.csv'))