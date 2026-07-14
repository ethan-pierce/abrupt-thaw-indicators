"""T37 follow-up — can a 1 km reproject SERVE native terrain values?

Decision (2026-07-14): the prediction surface upscales 4 km -> 1 km and terrain
is served by **native center-point sampling** (the datacube pixel carries the
native terrain value at its centre, matching the point path's
reduceRegion(mean, scale=native)).

The efficient way to do that in the datacube is the existing
``reproject(1 km) -> sampleRectangle`` array read — BUT only if EE can be made to
*interpolate* native values to the 1 km grid rather than *pyramid-aggregate*
them (the default, which flattened slope to ~0.28x native in probe_terrain_scale).
``image.resample('bilinear'|'bicubic')`` is supposed to switch reprojection from
aggregation to interpolation (the curvature port already relies on this). This
probe checks whether it actually recovers the native value.

For each layer, at ~200 points, compare the native point-path value against the
1 km reproject sampled four ways:
  * ``def4km``   — reproject(4 km), no resample   (the current datacube; baseline)
  * ``def1km``   — reproject(1 km), no resample   (just upscaling, still aggregates)
  * ``bilin1km`` — resample('bilinear') then reproject(1 km)
  * ``bicub1km`` — resample('bicubic')  then reproject(1 km)

Whichever column tracks ``native`` (ratio ~1, corr ~1) is the serving method to
wire into build_prediction_data.py. If none do, we fall back to literal
cell-centre point sampling.

Run: ``poetry run python diagnostics/probe_native_serve.py``
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
CRS = 'EPSG:4326'

elevation = ee.Image('USGS/3DEP/10m').select('elevation')
LAYERS = [
    ('slope',     lambda: ee.Terrain.slope(elevation),                              10),
    ('curv_500m', lambda: gee_features.mean_curvature(500).select('MeanCurvature'),  250),
    ('curv_2km',  lambda: gee_features.mean_curvature(2000).select('MeanCurvature'), 1000),
]


def main():
    pts = pd.read_csv(DATA / 'Alaska_Permafrost_Thaw_Database_v2.0.0.csv',
                      sep=',', encoding='latin1')
    samp = pts.sample(n=N_POINTS, random_state=SEED)[['Longitude', 'Latitude']]
    features = [ee.Feature(ee.Geometry.Point([float(lon), float(lat)]))
                for lon, lat in zip(samp['Longitude'], samp['Latitude'])]
    fc = ee.FeatureCollection(features)

    print(f'Native-serve probe: {N_POINTS} points, seed {SEED}\n')

    for label, builder, native_scale in LAYERS:
        img = builder()
        variants = {
            'native':   (img, native_scale),
            'def4km':   (img.reproject(crs=CRS, scale=4000), 4000),
            'def1km':   (img.reproject(crs=CRS, scale=1000), 1000),
            'bilin1km': (img.resample('bilinear').reproject(crs=CRS, scale=1000), 1000),
            'bicub1km': (img.resample('bicubic').reproject(crs=CRS, scale=1000), 1000),
        }

        def sampler(feat, variants=variants):
            pt = feat.geometry()
            out = {}
            for key, (im, sc) in variants.items():
                red = ee.Reducer.mean() if key == 'native' else ee.Reducer.first()
                out[key] = im.rename('v').reduceRegion(red, pt, sc, CRS).get('v')
            return feat.set(out)

        df = ee.data.computeFeatures({
            'expression': fc.map(sampler), 'fileFormat': 'PANDAS_DATAFRAME'})
        cols = list(variants.keys())
        df = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
        nat = df['native'].to_numpy()
        mask = np.abs(nat) > 1e-9

        print(f'== {label} (native scale {native_scale} m, n={len(df)}) ==')
        print(f'   {"variant":9s}  {"corr":>6s}  {"med|v|/|nat|":>12s}  {"RMSE":>10s}')
        for key in cols:
            v = df[key].to_numpy()
            corr = np.corrcoef(nat, v)[0, 1] if len(v) > 2 else np.nan
            ratio = np.median(np.abs(v[mask]) / np.abs(nat[mask])) if mask.any() else np.nan
            rmse = float(np.sqrt(np.mean((v - nat) ** 2)))
            flag = '  <- recovers native' if (key != 'native' and corr > 0.95
                                              and 0.9 < ratio < 1.1) else ''
            print(f'   {key:9s}  {corr:6.3f}  {ratio:12.3f}  {rmse:10.5g}{flag}')
        print()

    print('Pick the cheapest variant that recovers native (corr ~1, ratio ~1) as '
          'the datacube terrain serving method. If only literal point-sampling '
          'matches, wire that instead.')


if __name__ == '__main__':
    main()
