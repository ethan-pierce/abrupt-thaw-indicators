"""Load the model-input matrix with each row's Latitude/Longitude kept as metadata.

Mirrors `data/clean_feature_table.py` and asserts the result matches
`features_clean.csv`. Coordinates are never returned inside X.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA

LAND_COVER_LABELS = {
    0: 'NaN', 11: 'Open Water', 12: 'Perennial Ice/Snow', 21: 'Developed, Open Space',
    22: 'Developed, Low Intensity', 23: 'Developed, Medium Intensity',
    24: 'Developed, High Intensity', 31: 'Barren Land (Rock/Sand/Clay)',
    41: 'Deciduous Forest', 42: 'Evergreen Forest', 43: 'Mixed Forest',
    51: 'Dwarf Scrub', 52: 'Shrub/Scrub', 71: 'Grassland/Herbaceous',
    72: 'Sedge/Herbaceous', 73: 'Lichens', 74: 'Moss', 81: 'Pasture/Hay',
    82: 'Cultivated Crops', 90: 'Woody Wetlands', 95: 'Emergent Herbaceous Wetlands',
}
VEGETATION_MODE_LABELS = {
    0: 'NaN', 1: 'Black spruce', 2: 'White spruce', 3: 'Deciduous forest',
    4: 'Shrub tundra', 5: 'Graminoid tundra', 6: 'Wetland tundra',
    7: 'Barren lichen moss', 8: 'Temperate rainforest',
}
DROP_TEXT = ['Authors', 'DOI', 'DataSourceType', 'FeatureName', 'FeatureType',
             'FeatureCategory', 'Imagery', 'ImageryDates', 'ImageryResolution_meters']


def _clean_with_coords():
    """Replicate clean_feature_table.py, carrying Latitude/Longitude as metadata."""
    feats = pd.read_csv(DATA / 'features_dirty.csv')
    feats['Class'] = np.where(feats['ThawType'] == 'Abrupt', 0, 1)
    feats = feats.drop(['ThawType'] + DROP_TEXT, axis=1)

    for _snap in ['Projected summer temperature change', 'Projected winter temperature change',
                  'Projected precipitation change']:
        if _snap in feats.columns:
            feats = feats.drop(_snap, axis=1)

    for col, labels in [('Land Cover', LAND_COVER_LABELS), ('Vegetation Mode', VEGETATION_MODE_LABELS)]:
        for cat in feats[col].unique():
            if col == 'Vegetation Mode' and (isinstance(cat, float) and np.isnan(cat)):
                continue
            feats[f'{col} ({labels[cat]})'] = np.where(feats[col] == cat, 1, 0)
        feats = feats.drop(col, axis=1)

    for v in ['Soil Organic Carbon', 'Nitrogen', 'Bulk Density', 'Sand', 'Silt', 'Clay']:
        feats[f'{v} (0-30 cm)'] = (1/30) * (feats[f'{v} (0-5 cm)']*5 + feats[f'{v} (5-15 cm)']*10 + feats[f'{v} (15-30 cm)']*15)
        feats[f'{v} (30-200 cm)'] = (1/170) * (feats[f'{v} (30-60 cm)']*30 + feats[f'{v} (60-100 cm)']*40 + feats[f'{v} (100-200 cm)']*100)
        for d in ['0-5 cm', '5-15 cm', '15-30 cm', '30-60 cm', '60-100 cm', '100-200 cm']:
            feats.drop(f'{v} ({d})', axis=1, inplace=True)

    for d in ['0-30 cm', '30-200 cm']:
        feats.drop(f'Silt ({d})', axis=1, inplace=True)

    for c in ['Land Cover (NaN)', 'Vegetation Mode (NaN)']:
        if c in feats.columns:
            feats.drop(c, axis=1, inplace=True)

    # Dedup on feature + Class columns, as the pipeline does.
    coords = feats[['Latitude', 'Longitude']].copy()
    feats = feats.drop(['Longitude', 'Latitude'], axis=1)
    dedup_mask = ~feats.duplicated(keep='first')
    feats = feats[dedup_mask].reset_index(drop=True)
    coords = coords[dedup_mask].reset_index(drop=True)
    return feats, coords


def load(verify=True):
    """Return (X, y, lat, lon), with X restricted to the columns in features_clean.csv."""
    feats, coords = _clean_with_coords()
    clean = pd.read_csv(DATA / 'features_clean.csv')

    META = {'Latitude', 'Longitude'}

    if verify:
        assert len(feats) == len(clean), f"row mismatch: recon {len(feats)} vs clean {len(clean)}"
        assert (feats['Class'].values == clean['Class'].values).all(), "Class column mismatch"
        missing = set(clean.columns) - set(feats.columns) - META
        assert not missing, f"clean has columns the reconstruction lacks: {missing}"
        for c in clean.columns:
            if c in META:
                continue
            if not np.allclose(feats[c].values, clean[c].values, equal_nan=True):
                raise AssertionError(f"value mismatch in column {c!r}")
        for c in META & set(clean.columns):
            recon_coord = coords['Latitude' if c == 'Latitude' else 'Longitude'].values
            if not np.allclose(recon_coord, clean[c].values, equal_nan=True):
                raise AssertionError(f"coordinate mismatch in column {c!r}")

    feature_cols = [c for c in clean.columns if c not in ({'Class'} | META)]
    X = feats[feature_cols].copy()
    y = feats['Class'].astype(int).copy()
    return X, y, coords['Latitude'].values, coords['Longitude'].values


if __name__ == '__main__':
    X, y, lat, lon = load(verify=True)
    print(f"reconstruction OK: X={X.shape}, class balance={dict(y.value_counts())}")
    print(f"coords present for all rows: lat {np.isfinite(lat).all()}, lon {np.isfinite(lon).all()}")
    print(f"lat range [{lat.min():.3f}, {lat.max():.3f}], lon range [{lon.min():.3f}, {lon.max():.3f}]")
