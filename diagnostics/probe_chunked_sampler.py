"""Validate the chunked native cell-centre sampler (T37 datacube serving).

Checks that ee_sampling.sample_points_reduceregions_chunked, used by
build_prediction_data.sample_native, returns exactly what the point path's
reduceRegion produces at the same coordinates (train/serve parity by
construction), handles off-grid coords as NaN, and that the T32 northness/
eastness + flats math behaves. Small N so it runs in seconds.
"""

import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data'))
from settings import DATA, EE_PROJECT
import ee_sampling

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)

elevation = ee.Image('USGS/3DEP/10m').select('elevation')
slope = ee.Terrain.slope(elevation)
aspect = ee.Terrain.aspect(elevation)

# A handful of ThawDB points, plus two deliberately-invalid coords (the -9999
# off-ROI fill the datacube grid carries) to confirm they map to NaN.
pts = pd.read_csv(DATA / 'Alaska_Permafrost_Thaw_Database_v2.0.0.csv',
                  sep=',', encoding='latin1').sample(n=30, random_state=1)
lons = np.concatenate([pts['Longitude'].to_numpy(), [-9999.0, -9999.0]])
lats = np.concatenate([pts['Latitude'].to_numpy(), [-9999.0, -9999.0]])

print(f'chunked sampler validation: {lons.size} coords (2 invalid), chunk=8\n')

# Chunked serve-path value...
serve = ee_sampling.sample_points_reduceregions_chunked(
    lons, lats, slope, ee.Reducer.mean(), 10, 'slope', chunk=8)

# ...vs the point-path construction (single reduceRegions over the valid coords).
ok = (np.abs(lons) <= 180) & (np.abs(lats) <= 90)
ref = ee_sampling.sample_points_reduceregions(
    lons[ok], lats[ok], slope, ee.Reducer.mean(), 10, 'slope')

serve_valid = serve[ok]
assert np.isnan(serve[~ok]).all(), 'invalid coords did not map to NaN'
both = np.isfinite(serve_valid) & np.isfinite(ref)
maxdiff = float(np.max(np.abs(serve_valid[both] - ref[both]))) if both.any() else 0.0
print(f'  valid coords: {ok.sum()}, invalid->NaN: {(~ok).sum()} (all NaN: OK)')
print(f'  chunked vs single-call slope max |diff| = {maxdiff:.3g} '
      f'(expect ~0 — identical construction)')
print(f'  slope range served: {np.nanmin(serve_valid):.3f}..{np.nanmax(serve_valid):.3f} deg')
assert maxdiff < 1e-6, 'chunked and single-call disagree — parity broken'

# T32 math sanity: cos/sin + flats neutralization.
asp = ee_sampling.sample_points_reduceregions_chunked(
    lons, lats, aspect, ee.Reducer.mean(), 10, 'aspect', chunk=8)[ok]
asp_rad = np.deg2rad(asp)
flat = serve_valid < 1.0
north = np.cos(asp_rad); north[flat] = 0.0
east = np.sin(asp_rad); east[flat] = 0.0
r = np.hypot(north[~flat & np.isfinite(north)], east[~flat & np.isfinite(east)])
print(f'\n  T32: {flat.sum()} flats (slope<1) neutralized to 0; '
      f'non-flat cos^2+sin^2 in [{r.min():.3f},{r.max():.3f}] (expect ~1)')
assert np.allclose(r, 1.0, atol=1e-6), 'northness^2+eastness^2 != 1 off flats'
print('\nAll checks passed.')
