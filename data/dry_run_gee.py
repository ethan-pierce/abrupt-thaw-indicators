"""Pre-build GEE dry-run + coverage report (T39 req 2).

Purpose
-------
Run this right before launching the ~12 h ``build_feature_table.py`` point run.
It exercises the SAME GEE compute the build uses, but over a few hundred points
spread STATEWIDE across the datacube ROI (not the training points), so it:

  1. validates **auth / band names / schema** for the inline GEE features —
     3DEP terrain (the T37 native-scale probe: elevation/slope/aspect/curvature),
     MERIT Hydro (hnd + upstream area, T34), a representative WorldClim bioclim
     band, and one SoilGrids band per property; a probe that RAISES is a real
     auth/band/schema fault -> the gate fails; and
  2. reports **statewide NaN fractions for terrain and soil** — the empirical
     half of the 3DEP / SoilGrids coverage caveat (SCOPE). Terrain and soil are
     legitimately NaN over parts of the ROI (3DEP gaps; SoilGrids masked over
     water/ice/rock), so here NaN is *measured and reported*, not failed.

It also verifies the LOCAL-track sources the build reads at the end exist and
returns their statewide coverage, so the build's LOCAL block (which runs after
all the GEE work) can't surprise you with a missing raster after hours of work.

Pass / fail
-----------
Exit 0 iff (a) no GEE probe raised, and (b) every REQUIRED local source exists.
NaN fractions are informational (reported, never failing) except an all-NaN GEE
probe, which is flagged as a likely wiring fault. The MODIS fire raster is
REQUIRED (build it first with build_modis_fire_rasters.py); its absence fails the
gate loudly, because the build's LOCAL track would otherwise fail on it at the
very end.

Run:
    poetry run python data/dry_run_gee.py [N]     # N statewide points, default 400
"""

import json
import sys
from pathlib import Path

import numpy as np
import ee
import geemap

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'data'))
from settings import DATA, EE_PROJECT
import gee_features as gf
import local_rasters
import build_daymet_rasters
import build_modis_fire_rasters

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400

# Non-interactive-safe init (mirrors build_feature_table.py): a valid cached token
# never triggers a browser prompt; ee.Authenticate() is a fallback only.
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

# --------------------------------------------------------------------------
# N points spread across the datacube ROI (deterministic seed). Materialize the
# coordinates once (getInfo) and rebuild a fixed FeatureCollection so every probe
# samples the identical points and the LOCAL rasters can be sampled at the same
# coordinates.
# --------------------------------------------------------------------------
with open(DATA / 'roi.geojson') as f:
    roi = geemap.geojson_to_ee(json.load(f)).geometry()
rand = ee.FeatureCollection.randomPoints(region=roi, points=N, seed=39)
mp = rand.geometry(maxError=1).coordinates().getInfo()   # [[lon, lat], ...]
lons = np.array([c[0] for c in mp], dtype=float)
lats = np.array([c[1] for c in mp], dtype=float)
N = len(lons)
points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(c)) for c in mp])
print(f'Dry-run: {N} points spread across the ROI; EE project {EE_PROJECT}\n')

errored = []       # probes that raised (auth/band/schema fault)
gee_results = {}   # feature name -> np.ndarray aligned to the N points


def gee_sample(image, reducer, scale, name, band, crs='EPSG:4326'):
    """Mirror build_feature_table.add_feature: mean-reduce at each point via one
    computeFeatures, keyed positionally over the statewide points."""
    sampler = lambda feat: feat.set(image.reduceRegion(
        reducer=reducer, geometry=feat.geometry(proj=crs), scale=scale, crs=crs))
    df = ee.data.computeFeatures(
        {'expression': points.map(sampler), 'fileFormat': 'PANDAS_DATAFRAME'})
    gee_results[name] = np.array(
        [row.get(band, np.nan) for _, row in df.iterrows()], dtype=float)


def probe(name, fn):
    try:
        fn()
        print(f'  ok   {name}')
    except Exception as e:
        gee_results[name] = np.full(N, np.nan)
        errored.append(name)
        print(f'  ERR  {name}: {e!r}')


# ---- GEE inline: terrain (3DEP, T37 native-scale probe) ------------------
print('GEE inline — 3DEP terrain (T37 native scale):')
_dem = ee.Image(gf._DEM_ID).select('elevation')
probe('Elevation', lambda: gee_sample(_dem, ee.Reducer.mean(), 10, 'Elevation', 'elevation'))
probe('Slope', lambda: gee_sample(ee.Terrain.slope(_dem), ee.Reducer.mean(), 10, 'Slope', 'slope'))
probe('Aspect', lambda: gee_sample(ee.Terrain.aspect(_dem), ee.Reducer.mean(), 10, 'Aspect', 'aspect'))
probe('Mean curvature (500 m)', lambda: gee_sample(
    gf.mean_curvature(500), ee.Reducer.mean(), 250, 'Mean curvature (500 m)', 'MeanCurvature'))
probe('Mean curvature (2 km)', lambda: gee_sample(
    gf.mean_curvature(2000), ee.Reducer.mean(), 1000, 'Mean curvature (2 km)', 'MeanCurvature'))

# ---- GEE inline: MERIT Hydro (T34) ---------------------------------------
print('GEE inline — MERIT Hydro (hnd + upstream area):')
probe('Height Above Nearest Drainage', lambda: gee_sample(
    gf.height_above_drainage(), ee.Reducer.mean(), gf.MERIT_SCALE,
    'Height Above Nearest Drainage', 'hnd'))
probe('Upstream Area', lambda: gee_sample(
    gf.upstream_area(), ee.Reducer.mean(), gf.MERIT_SCALE, 'Upstream Area', 'upa'))

# ---- GEE inline: bioclim (representative band) ---------------------------
print('GEE inline — WorldClim bioclim (representative):')
_bioclim = ee.Image('WORLDCLIM/V1/BIO')
probe('Annual Mean Temperature', lambda: gee_sample(
    _bioclim, ee.Reducer.mean(), 1000, 'Annual Mean Temperature', 'bio01'))

# ---- GEE inline: SoilGrids (one band per property) -----------------------
print('GEE inline — SoilGrids (one band per property):')
for asset, band, name in (
        ('soc_mean', 'soc_0-5cm_mean', 'Soil Organic Carbon (0-5 cm)'),
        ('nitrogen_mean', 'nitrogen_0-5cm_mean', 'Nitrogen (0-5 cm)'),
        ('clay_mean', 'clay_0-5cm_mean', 'Clay (0-5 cm)'),
        ('sand_mean', 'sand_0-5cm_mean', 'Sand (0-5 cm)'),
        ('silt_mean', 'silt_0-5cm_mean', 'Silt (0-5 cm)'),
        ('bdod_mean', 'bdod_0-5cm_mean', 'Bulk Density (0-5 cm)')):
    img = ee.Image(f'projects/soilgrids-isric/{asset}')
    probe(name, lambda i=img, b=band, n=name: gee_sample(i, ee.Reducer.mean(), 250, n, b))

# --------------------------------------------------------------------------
# LOCAL-track sources: existence + statewide coverage at the same points.
# REQUIRED sources missing -> gate fails (the build reads them at the very end).
# --------------------------------------------------------------------------
print('\nLOCAL — source existence + statewide coverage:')
local_missing = []      # required sources that are absent
local_coverage = {}     # label -> (finite, N) or None if source missing


def local_probe(label, path, required, sampler):
    exists = Path(path).exists()
    if not exists:
        local_coverage[label] = None
        tag = 'MISSING (REQUIRED)' if required else 'missing (optional)'
        print(f'  {tag:<20} {label}  [{path}]')
        if required:
            local_missing.append(label)
        return
    try:
        arr = sampler()
        local_coverage[label] = (int(np.isfinite(arr).sum()), N)
        print(f'  ok                   {label}')
    except Exception as e:
        local_coverage[label] = (0, N)
        local_missing.append(label) if required else None
        print(f'  ERR                  {label}: {e!r}')


local_probe('Land Cover (NLCD)', local_rasters.NLCD_IMG, True,
            lambda: local_rasters.sample_points(local_rasters.NLCD_IMG, lons, lats))
local_probe('Vegetation Mode (ALFRESCO)', local_rasters.VEGMODE_TIF, True,
            lambda: local_rasters.sample_points(local_rasters.VEGMODE_TIF, lons, lats))
local_probe('Flammability Index (ALFRESCO)', local_rasters.FLAMMABILITY_TIF, True,
            lambda: local_rasters.sample_points(local_rasters.FLAMMABILITY_TIF, lons, lats))
for _feat, _band in local_rasters.DAYMET_BANDS.items():
    local_probe(f'{_feat} (Daymet)', local_rasters.DAYMET_TIF, True,
                lambda b=_band: local_rasters.sample_points(local_rasters.DAYMET_TIF, lons, lats, band=b))
for _feat, _band in local_rasters.MODIS_FIRE_BANDS.items():
    local_probe(f'{_feat} (MODIS)', local_rasters.MODIS_FIRE_TIF, True,
                lambda b=_band: local_rasters.sample_points(local_rasters.MODIS_FIRE_TIF, lons, lats, band=b))
local_probe('Yedoma (IRYP v2)', local_rasters.YEDOMA_SHP, True,
            lambda: local_rasters.sample_yedoma(lons, lats))

# --------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------
TERRAIN = ['Elevation', 'Slope', 'Aspect', 'Mean curvature (500 m)', 'Mean curvature (2 km)']
SOIL = ['Soil Organic Carbon (0-5 cm)', 'Nitrogen (0-5 cm)', 'Clay (0-5 cm)',
        'Sand (0-5 cm)', 'Silt (0-5 cm)', 'Bulk Density (0-5 cm)']

print('\n' + '=' * 74)
print(f'{"GEE feature":<34}{"finite/N":>12}{"NaN %":>10}{"sample":>16}')
print('=' * 74)
all_nan_gee = []
for name, arr in gee_results.items():
    finite = np.isfinite(arr)
    nfin = int(finite.sum())
    nanpct = 100.0 * (1 - nfin / N) if N else float('nan')
    ex = f'{arr[finite][0]:.4g}' if nfin else '--'
    note = '  <- ERROR' if name in errored else ('  <- all-NaN' if nfin == 0 else '')
    if nfin == 0 and name not in errored:
        all_nan_gee.append(name)
    print(f'{name:<34}{f"{nfin}/{N}":>12}{nanpct:>9.1f}%{ex:>16}{note}')
print('=' * 74)


def _group_nan(group):
    vals = [gee_results[n] for n in group if n in gee_results]
    if not vals:
        return float('nan')
    stacked = np.vstack(vals)
    return 100.0 * (1 - np.isfinite(stacked).mean())


print(f'\nStatewide coverage caveat (T39): '
      f'terrain NaN {_group_nan(TERRAIN):.1f}% | soil NaN {_group_nan(SOIL):.1f}% '
      f'(mean over the group, {N} ROI points).')

print('\nLOCAL-track coverage (finite / N):')
for label, cov in local_coverage.items():
    if cov is None:
        print(f'  {label:<40} MISSING')
    else:
        fin, n = cov
        print(f'  {label:<40} {fin}/{n}   ({100.0*fin/n:.1f}% finite)')

# --------------------------------------------------------------------------
# Gate.
# --------------------------------------------------------------------------
print('\n' + '=' * 74)
fail = False
if errored:
    fail = True
    print(f'FAIL: {len(errored)} GEE probe(s) raised (auth/band/schema fault):')
    for n in errored:
        print(f'  - {n}')
if all_nan_gee:
    fail = True
    print(f'FAIL: {len(all_nan_gee)} GEE probe(s) returned all-NaN (likely wiring fault):')
    for n in all_nan_gee:
        print(f'  - {n}')
if local_missing:
    fail = True
    print(f'FAIL: {len(local_missing)} required LOCAL source(s) missing/unreadable:')
    for n in local_missing:
        print(f'  - {n}')
if fail:
    print('=' * 74)
    print('\nDO NOT launch the build until the above are resolved.')
    sys.exit(1)
print('=' * 74)
print('\nPASS: GEE compute validated (auth/bands/schema), all required LOCAL sources '
      'present. Terrain/soil NaN fractions above are the expected coverage caveat. '
      'Safe to launch build_feature_table.py.')
