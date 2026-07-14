"""T37 probe — terrain train/serve scale: recompute vs. resample at 4 km.

Question (TASKS T37): the point path (build_feature_table.py) samples slope /
curvature at native scale (reduceRegion(mean, scale=10 / 250 / 1000)); the
datacube (build_prediction_data.py) extracts the SAME derived image after
``.reproject(EPSG:4326, scale=4000)``. Does Earth Engine's reproject at 4 km
**recompute** the derivative from a 4 km-resampled DEM (→ dramatically flatter
slopes / smoother curvature: a SEVERE train/serve mismatch), or **resample** the
already-computed native-scale derivative to the 4 km grid (→ ~native value at the
cell: a MILD mismatch)?

This probe settles it empirically over ~200 ThawDB points by sampling each
terrain derivative two ways and comparing the paired distributions:

  * ``native``  — reduceRegion(mean) at the analysis scale used by the point path.
  * ``at4km``   — the datacube construction verbatim: reproject the derived image
                  to EPSG:4326 @ 4000 m, then read the covering 4 km cell.

If ``at4km`` tracks ``native`` (ratio ~1, high correlation) the mismatch is a
resample and is mild → document and leave. If ``at4km`` collapses toward zero
(slope) or is systematically smoother (curvature) it is a recompute → the
coarsen-vs-matching-columns fix (T35) must be applied to BOTH paths before the
full build.

Run: ``poetry run python diagnostics/probe_terrain_scale.py``
"""

import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data'))
from settings import DATA, EE_PROJECT
import gee_features

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)

N_POINTS = 200
SEED = 37
DATACUBE_SCALE = 4000
DATACUBE_CRS = 'EPSG:4326'  # datacube reprojects elevation to EPSG:4326 @ 4 km

# (label, derived-image builder, native analysis scale used by the point path)
elevation = ee.Image('USGS/3DEP/10m').select('elevation')
LAYERS = [
    ('slope',      lambda: ee.Terrain.slope(elevation),               10),
    ('curv_500m',  lambda: gee_features.mean_curvature(500).select('MeanCurvature'),  250),
    ('curv_2km',   lambda: gee_features.mean_curvature(2000).select('MeanCurvature'), 1000),
]


def sample_pair(img: ee.Image, native_scale: float):
    """Return (native_reducer_result, datacube_4km_result) as two ee dicts keyed
    by band, for the point collection defined below."""
    band = img.bandNames().get(0)
    native = img.rename('native')
    at4km = (img.reproject(crs=DATACUBE_CRS, scale=DATACUBE_SCALE)
             .rename('at4km'))
    return native, at4km, native_scale


def main():
    pts = pd.read_csv(DATA / 'Alaska_Permafrost_Thaw_Database_v2.0.0.csv',
                      sep=',', encoding='latin1')
    samp = pts.sample(n=N_POINTS, random_state=SEED)[['Longitude', 'Latitude']]

    features = [ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {'idx': i})
                for i, (lon, lat) in enumerate(zip(samp['Longitude'], samp['Latitude']))]
    fc = ee.FeatureCollection(features)

    print(f'T37 terrain-scale probe: {N_POINTS} points, seed {SEED}\n')

    results = {}
    for label, builder, native_scale in LAYERS:
        img = builder()
        native = img.rename('v')
        at4km = img.reproject(crs=DATACUBE_CRS, scale=DATACUBE_SCALE).rename('v')

        def sampler(feat, native=native, at4km=at4km, ns=native_scale):
            pt = feat.geometry()
            nv = native.reduceRegion(ee.Reducer.mean(), pt, ns, DATACUBE_CRS).get('v')
            # datacube reads the covering 4 km cell (the reprojected pixel value):
            fv = at4km.reduceRegion(ee.Reducer.first(), pt, DATACUBE_SCALE,
                                    DATACUBE_CRS).get('v')
            return feat.set({'native': nv, 'at4km': fv})

        sampled = ee.data.computeFeatures({
            'expression': fc.map(sampler),
            'fileFormat': 'PANDAS_DATAFRAME',
        })
        df = sampled[['native', 'at4km']].apply(pd.to_numeric, errors='coerce').dropna()
        results[label] = df

        n, a = df['native'].to_numpy(), df['at4km'].to_numpy()
        corr = np.corrcoef(n, a)[0, 1] if len(n) > 2 else np.nan
        # ratio of |at4km| to |native| (avoid div-by-zero on near-flat curvature)
        mask = np.abs(n) > 1e-9
        ratio = np.median(np.abs(a[mask]) / np.abs(n[mask])) if mask.any() else np.nan
        rmse = float(np.sqrt(np.mean((a - n) ** 2)))
        print(f'== {label} (native scale {native_scale} m, n={len(df)}) ==')
        print(f'   native : mean {n.mean():+.5g}  std {n.std():.5g}  '
              f'|.| median {np.median(np.abs(n)):.5g}')
        print(f'   at4km  : mean {a.mean():+.5g}  std {a.std():.5g}  '
              f'|.| median {np.median(np.abs(a)):.5g}')
        print(f'   corr(native, at4km) = {corr:.3f}')
        print(f'   median |at4km|/|native| = {ratio:.3f}')
        print(f'   RMSE(at4km - native)    = {rmse:.5g}\n')

    print('Interpretation: ratio ~1 and high corr → RESAMPLE (mild, document & '
          'leave). ratio << 1 (esp. slope) or low corr → RECOMPUTE (severe, '
          'coarsen both paths per T35 before the build).')
    return results


if __name__ == '__main__':
    main()
