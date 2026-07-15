"""Shared loader for the verify-ml diagnostics suite.

Reconstructs the exact model-input matrix the training pipeline sees
(`features_clean.csv`) *while retaining each row's Latitude/Longitude* so the
spatial-leakage probes can group points by location. Coordinates are NEVER
returned inside X — they are metadata only, exactly as the pipeline intends.

The reconstruction mirrors `data/clean_feature_table.py` step for step and then
asserts it matches the committed `features_clean.csv` (row count + class
balance + feature-column set), so the probes are provably testing the real
pipeline and not a look-alike.
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

    # SNAP projected-climate features removed 2026-07-13 (see PIPELINE.md); drop
    # defensively so a stale dirty table can't reintroduce them, matching clean_feature_table.py.
    for _snap in ['Projected summer temperature change', 'Projected winter temperature change',
                  'Projected precipitation change']:
        if _snap in feats.columns:
            feats = feats.drop(_snap, axis=1)

    # T36: fire is now the MODIS MCD64A1 history pair (Time Since Last Fire, Burn
    # Count), continuous columns that pass through untouched — no fill, matching
    # clean_feature_table.py (the retired FIRMS Maximum Fire Temperature fill is gone).

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

    # Compositional closure: drop Silt, keep Sand + Clay, exactly as clean_feature_table.py (T35).
    for d in ['0-30 cm', '30-200 cm']:
        feats.drop(f'Silt ({d})', axis=1, inplace=True)

    for c in ['Land Cover (NaN)', 'Vegetation Mode (NaN)']:
        if c in feats.columns:
            feats.drop(c, axis=1, inplace=True)

    # Keep coords aside, dedup on feature+Class columns exactly as the pipeline does.
    coords = feats[['Latitude', 'Longitude']].copy()
    feats = feats.drop(['Longitude', 'Latitude'], axis=1)
    dedup_mask = ~feats.duplicated(keep='first')
    feats = feats[dedup_mask].reset_index(drop=True)
    coords = coords[dedup_mask].reset_index(drop=True)
    return feats, coords


def load(verify=True):
    """Return (X, y, lat, lon) — X/y identical to the pipeline; lat/lon are metadata.

    The current features_clean.csv may carry a *pared* subset of the columns this
    reconstruction produces (the full-set restore is applied at retrain time). We
    therefore align X to whatever columns features_clean.csv actually contains, and
    assert row-for-row equality on those, so the probes test the live model input.
    """
    feats, coords = _clean_with_coords()
    clean = pd.read_csv(DATA / 'features_clean.csv')

    if verify:
        assert len(feats) == len(clean), f"row mismatch: recon {len(feats)} vs clean {len(clean)}"
        assert (feats['Class'].values == clean['Class'].values).all(), "Class column mismatch"
        missing = set(clean.columns) - set(feats.columns)
        assert not missing, f"clean has columns the reconstruction lacks: {missing}"
        for c in clean.columns:
            if not np.allclose(feats[c].values, clean[c].values, equal_nan=True):
                raise AssertionError(f"value mismatch in column {c!r}")

    # Use the pipeline's own column set (features_clean minus Class).
    feature_cols = [c for c in clean.columns if c != 'Class']
    X = feats[feature_cols].copy()
    y = feats['Class'].astype(int).copy()
    return X, y, coords['Latitude'].values, coords['Longitude'].values


if __name__ == '__main__':
    X, y, lat, lon = load(verify=True)
    print(f"reconstruction OK: X={X.shape}, class balance={dict(y.value_counts())}")
    print(f"coords present for all rows: lat {np.isfinite(lat).all()}, lon {np.isfinite(lon).all()}")
    print(f"lat range [{lat.min():.3f}, {lat.max():.3f}], lon range [{lon.min():.3f}, {lon.max():.3f}]")
