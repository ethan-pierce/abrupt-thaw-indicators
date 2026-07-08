"""feats the feature table."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA

feats = pd.read_csv(DATA / 'features_dirty.csv')
feats['Class'] = np.where(feats['ThawType'] == 'Abrupt', 0, 1)  # Abrupt = 0 (majority class), Gradual = 1 (minority class)
feats = feats.drop('ThawType', axis = 1)
feats = feats.drop('Authors', axis = 1)
feats = feats.drop('DOI', axis = 1)
feats = feats.drop('DataSourceType', axis = 1)
feats = feats.drop('FeatureName', axis = 1)
feats = feats.drop('FeatureType', axis = 1)
feats = feats.drop('FeatureCategory', axis = 1)
feats = feats.drop('Imagery', axis = 1)
feats = feats.drop('ImageryDates', axis = 1)
feats = feats.drop('ImageryResolution_meters', axis = 1)

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

for col in feats.columns:
    if col in label:
        continue
    if col in fillna:
        feats[col] = np.where(np.isnan(feats[col]), 0.0, feats[col])
    if col in categorical:
        categories = feats[col].unique()
        for cat in categories:
            if col == 'Land Cover':
                feats[col + ' (' + land_cover_labels[cat] + ')'] = np.where(feats[col] == cat, 1, 0)
            if col == 'Vegetation Mode':
                if np.isnan(cat):
                    continue
                feats[col + ' (' + vegetation_mode_labels[cat] + ')'] = np.where(feats[col] == cat, 1, 0)
        feats = feats.drop(col, axis=1)
    else:
        continue

for variable in ['Soil Organic Carbon', 'Nitrogen', 'Bulk Density', 'Sand', 'Silt', 'Clay']:
    feats[variable + ' (0-30 cm)'] = (1 / 30) * (
        feats[variable + ' (0-5 cm)'] * 5 + feats[variable + ' (5-15 cm)'] * 10 + feats[variable + ' (15-30 cm)'] * 15
    )
    feats[variable + ' (30-200 cm)'] = (1 / 170) * (
        feats[variable + ' (30-60 cm)'] * 30 + feats[variable + ' (60-100 cm)'] * 40 + feats[variable + ' (100-200 cm)'] * 100
    )

    for depth in ['0-5 cm', '5-15 cm', '15-30 cm', '30-60 cm', '60-100 cm', '100-200 cm']:
        feats.drop(variable + ' (' + depth + ')', axis = 1, inplace = True)
    
# If preparing for XGBoost, no need to drop NaN values
# Unless using SMOTE
# feats = feats.dropna(axis = 0, how = 'any')

# Drop only the NaN one-hot columns — structural cleanup, not feature selection.
# Retrain #1 keeps the full feature set; rigorous paring (VIF / collinearity /
# coverage) is a separate documented protocol applied afterward (README to-do #15).
if 'Land Cover (NaN)' in feats.columns:
    feats.drop('Land Cover (NaN)', axis = 1, inplace = True)
if 'Vegetation Mode (NaN)' in feats.columns:
    feats.drop('Vegetation Mode (NaN)', axis = 1, inplace = True)

# Drop Longitude/Latitude from the feature set. This is unconditional (NOT optional):
# raw coordinates are excluded to avoid spatial leakage, and removing them is also
# what lets near-identical points collapse under drop_duplicates.
feats = feats.drop(['Longitude', 'Latitude'], axis = 1)
print('Duplicate rows:', feats.duplicated().sum())
feats = feats.drop_duplicates(keep = 'first')

print(feats['Class'].value_counts())
print(feats.shape)

feats.to_csv(DATA / 'features_clean.csv', index = False)