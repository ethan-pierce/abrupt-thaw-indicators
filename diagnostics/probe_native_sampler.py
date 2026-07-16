"""Validate the tiled multiband native cell-centre sampler (T47 datacube serving).

Checks that ee_sampling.sample_native_multiband_tiled — which serves the
datacube's native-scale terrain/soil by one bounded-footprint reduceRegions per
compact grid tile, merging same-scale bands into a single pass — returns what the
point path's single-call reduceRegions produces at the same coordinates
(train/serve parity by construction), that merging bands does not perturb values,
that off-grid coords (the -9999 datacube fill) map to NaN, and that the T32
northness/eastness + flats math behaves. Small grid so it runs in seconds.

Replaces probe_chunked_sampler.py: the index-chunked sampler it exercised was
removed under T47 (its chunks spanned the whole state at statewide scale).
"""

import sys
from pathlib import Path

import ee
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data'))
from settings import EE_PROJECT
import ee_sampling

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)

elevation = ee.Image('USGS/3DEP/10m').select('elevation')
slope = ee.Terrain.slope(elevation)
aspect = ee.Terrain.aspect(elevation)

# Small interior grid (near Fairbanks, all land), with two cells set to the -9999
# off-ROI fill the datacube grid carries, to confirm they sample to NaN.
lon2d, lat2d = np.meshgrid(np.linspace(-149.5, -148.5, 40),
                           np.linspace(65.5, 64.5, 50))
lon2d[0, 0] = lon2d[10, 5] = -9999.0
lat2d[0, 0] = lat2d[10, 5] = -9999.0
off = (np.abs(lon2d) > 180) | (np.abs(lat2d) > 90)

# tile=16 (< grid) so >1 tile is stitched -> reassembly-across-tiles is exercised.
multi = elevation.rename('elevation').addBands(slope.rename('slope')).addBands(aspect.rename('aspect'))
new = ee_sampling.sample_native_multiband_tiled(
    lon2d, lat2d, multi, ['elevation', 'slope', 'aspect'], 10, tile=16, workers=8)

print(f'native sampler validation: {lon2d.size} cells ({off.sum()} off-grid)\n')

# Off-grid cells -> NaN in every band.
for b in ('elevation', 'slope', 'aspect'):
    assert np.isnan(new[b][off]).all(), f'{b}: off-grid cells did not map to NaN'
print(f'  off-grid -> NaN in all bands: OK ({off.sum()} cells)')

# Merged/tiled parity vs the single-call point-path construction, per band.
# Tolerance, not bit-equality: elevation (raw) is exact; slope/aspect are
# recomputed on a reprojected DEM whose tile boundaries shift with the request
# footprint, giving sub-microdegree float noise (GEE-side, methodological parity).
for b, img in (('elevation', elevation), ('slope', slope), ('aspect', aspect)):
    ref = ee_sampling.sample_points_reduceregions(
        lon2d.ravel(), lat2d.ravel(), img, ee.Reducer.mean(), 10, b).reshape(lon2d.shape)
    nan_match = np.array_equal(np.isnan(ref), np.isnan(new[b]))
    both = np.isfinite(ref) & np.isfinite(new[b])
    maxdiff = float(np.max(np.abs(ref[both] - new[b][both]))) if both.any() else 0.0
    print(f'  {b:9s} tiled-multiband vs single-call: nan_match={nan_match} '
          f'max|diff|={maxdiff:.3g}')
    assert nan_match, f'{b}: NaN masks differ'
    assert maxdiff < 1e-4, f'{b}: values disagree beyond float noise — parity broken'

# Single-band group: reduceRegions names a lone mean-output column 'mean', NOT
# the band, so the reassembly must fall back to the sole numeric column or the
# feature ships all-NaN. Guard that regression (a single-band native scale group
# arises whenever a feature set leaves exactly one band at a scale, e.g. only
# Elevation at 10 m). Elevation is a raw read -> exact vs the single-call ref.
solo = ee_sampling.sample_native_multiband_tiled(
    lon2d, lat2d, elevation.rename('elevation'), ['elevation'], 10, tile=16, workers=8)
ref_elev = ee_sampling.sample_points_reduceregions(
    lon2d.ravel(), lat2d.ravel(), elevation, ee.Reducer.mean(), 10, 'elevation').reshape(lon2d.shape)
on = np.isfinite(ref_elev)
assert on.any() and np.isfinite(solo['elevation'][on]).all(), \
    'single-band group returned all-NaN — reduceRegions "mean"-column fallback missing'
assert np.array_equal(np.isnan(ref_elev), np.isnan(solo['elevation'])), 'single-band NaN mask differs'
assert np.max(np.abs(ref_elev[on] - solo['elevation'][on])) == 0, 'single-band values differ (raw read should be exact)'
print(f'\n  single-band group: elevation finite {100*on.mean():.1f}%, exact vs single-call: OK')

# T32 math sanity: cos/sin + flats neutralization off the sampled arrays.
valid = np.isfinite(new['slope'])
asp_rad = np.deg2rad(new['aspect'][valid])
sl = new['slope'][valid]
flat = sl < 1.0
north = np.cos(asp_rad); north[flat] = 0.0
east = np.sin(asp_rad); east[flat] = 0.0
r = np.hypot(north[~flat], east[~flat])
print(f'\n  T32: {flat.sum()} flats (slope<1) neutralized to 0; '
      f'non-flat cos^2+sin^2 in [{r.min():.3f},{r.max():.3f}] (expect ~1)')
assert np.allclose(r, 1.0, atol=1e-6), 'northness^2+eastness^2 != 1 off flats'
print('\nAll checks passed.')
