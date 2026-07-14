"""Pre-flight wiring smoke test for the feature build (T30).

Purpose
-------
The full ``build_feature_table.py`` run is ~12 h. This script exercises the SAME
feature-sourcing entry points at a tiny, evenly-spaced sample of real ThawDB
v2.0.0 points (default N=12) so import errors, wrong band names, bad keying, or a
missing local raster surface in ~minutes instead of after hours. It is a GATE to
run right before launching the real build — not part of the build itself.

Scope
-----
Read-only w.r.t. the builder modules: it imports ``gee_features``,
``local_rasters``, ``ee_sampling``, and ``build_daymet_rasters`` and calls their
public entry points; it never edits or runs the full builders. It samples a
REPRESENTATIVE subset of features (one probe per source image / track), which is
enough to catch wiring/keying faults — it is not full-column coverage.

Tracks covered:
  * GEE inline   : 3DEP elevation/slope/aspect, mean curvature (500 m & 2 km),
                   3 WorldClim bioclim bands, one SoilGrids band per property.
  * GEE reduce   : FIRMS max fire temp via ee_sampling.add_feature_reduceregions @ 4 km.
  * LOCAL        : NLCD land cover, ALFRESCO vegetation mode + flammability.
  * Daymet (SWE/trends): CONDITIONAL — validated straight off the materialized
                   raster (build_daymet_rasters.OUT_TIF, 4 bands) once it exists.
                   Reported as PENDING (soft-skip) until the other agent's
                   migration produces it; never fails the gate while pending.

Pass / fail
-----------
Exit 0 iff every non-pending probed feature either returns >=1 finite value at the
sample points, or is an EXPECTED-SPARSE feature (see SPARSE) whose probe call
succeeded but is legitimately all-NaN at the small sample (e.g. FIRMS max fire
temp, masked wherever no fire was ever detected). A probe that RAISES, or an
unexpected all-NaN in a non-sparse feature (a real wiring fault) -> exit 1.
Daymet PENDING is a soft skip (does not fail); once the raster exists it becomes
a hard probe like the rest.

Run (only after the Daymet migration lands):
    poetry run python data/smoke_feature_build.py [N]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ee

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'data'))
from settings import DATA, EE_PROJECT
import gee_features as gf
import local_rasters
import ee_sampling
import build_daymet_rasters

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

ee.Authenticate()
ee.Initialize(project=EE_PROJECT)

# --------------------------------------------------------------------------
# Evenly-spaced real ThawDB v2.0.0 points (deterministic — no RNG).
# --------------------------------------------------------------------------
thawdb = pd.read_csv(DATA / 'Alaska_Permafrost_Thaw_Database_v2.0.0.csv',
                     sep=',', encoding='latin1')
idx = np.linspace(0, len(thawdb) - 1, N).astype(int)
sub = thawdb.iloc[idx].reset_index(drop=True).copy()
lons, lats = sub['Longitude'].to_numpy(), sub['Latitude'].to_numpy()
points = ee.FeatureCollection(
    [ee.Feature(ee.Geometry.Point([lo, la])) for lo, la in zip(lons, lats)])
print(f'Smoke: {N} evenly-spaced ThawDB v2.0.0 points; EE project {EE_PROJECT}\n')

results = {}   # feature name -> np.ndarray aligned to the N points
pending = []   # soft-skipped feature names (Daymet, until migration lands)
errored = []   # features whose probe call raised (a real fault)

# Expected-sparse features: legitimately all-NaN at a small point sample because
# they are masked over most of Alaska. FIRMS max fire temp is masked wherever no
# fire was ever detected (~5% of ThawDB points ever burned), so 0 finite at N~12
# is the likely outcome and must NOT fail the gate — provided the probe SUCCEEDED
# (a raised probe is still a real fault and fails regardless).
SPARSE = {'Maximum Fire Temperature'}


def add_feature_gee(image, reducer, scale, name, band, crs='EPSG:4326'):
    """Mirror build_feature_table.add_feature: mean-reduce at each point via a
    single computeFeatures, keyed positionally over the small subset."""
    sampler = lambda feat: feat.set(image.reduceRegion(
        reducer=reducer, geometry=feat.geometry(proj=crs), scale=scale, crs=crs))
    df = ee.data.computeFeatures(
        {'expression': points.map(sampler), 'fileFormat': 'PANDAS_DATAFRAME'})
    results[name] = np.array([row.get(band, np.nan) for _, row in df.iterrows()],
                             dtype=float)


def probe(fn, label):
    try:
        fn()
        print(f'  ok   {label}')
    except Exception as e:
        results[label] = np.full(N, np.nan)
        errored.append(label)
        print(f'  ERR  {label}: {e!r}')


# ---- GEE inline: terrain -------------------------------------------------
print('GEE inline — terrain (3DEP):')
dem = ee.Image(gf._DEM_ID).select('elevation')
probe(lambda: add_feature_gee(dem, ee.Reducer.mean(), 10, 'Elevation', 'elevation'), 'Elevation')
probe(lambda: add_feature_gee(ee.Terrain.slope(dem), ee.Reducer.mean(), 10, 'Slope', 'slope'), 'Slope')
probe(lambda: add_feature_gee(ee.Terrain.aspect(dem), ee.Reducer.mean(), 10, 'Aspect', 'aspect'), 'Aspect')
probe(lambda: add_feature_gee(gf.mean_curvature(500), ee.Reducer.mean(), 250,
                              'Mean curvature (500 m)', 'MeanCurvature'), 'Mean curvature (500 m)')
probe(lambda: add_feature_gee(gf.mean_curvature(2000), ee.Reducer.mean(), 1000,
                              'Mean curvature (2 km)', 'MeanCurvature'), 'Mean curvature (2 km)')

# ---- GEE inline: hydrological terrain (MERIT Hydro, T34) -----------------
print('GEE inline — MERIT Hydro (hnd + log upstream area):')
probe(lambda: add_feature_gee(gf.height_above_drainage(), ee.Reducer.mean(),
                              gf.MERIT_SCALE, 'Height Above Nearest Drainage', 'hnd'),
      'Height Above Nearest Drainage')
probe(lambda: add_feature_gee(gf.log_upstream_area(), ee.Reducer.mean(),
                              gf.MERIT_SCALE, 'Log Upstream Area', 'log_upa'),
      'Log Upstream Area')

# ---- GEE inline: bioclim (representative bands) --------------------------
print('GEE inline — WorldClim bioclim (representative):')
bioclim = ee.Image('WORLDCLIM/V1/BIO')
for band, name in (('bio01', 'Annual Mean Temperature'),
                   ('bio04', 'Temperature Seasonality'),
                   ('bio12', 'Annual Precipitation')):
    probe(lambda b=band, n=name: add_feature_gee(bioclim, ee.Reducer.mean(), 1000, n, b), name)

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
    probe(lambda i=img, b=band, n=name: add_feature_gee(i, ee.Reducer.mean(), 250, n, b), name)

# ---- GEE reduce: FIRMS via ee_sampling @ 4 km ----------------------------
print('GEE reduceRegions — FIRMS max fire temp @ 4 km:')
probe(lambda: ee_sampling.add_feature_reduceregions(
    sub, lons, lats, gf.max_fire_temp(), ee.Reducer.mean(), 4000,
    'Maximum Fire Temperature', 'T21') or results.__setitem__(
    'Maximum Fire Temperature', sub['Maximum Fire Temperature'].to_numpy()),
    'Maximum Fire Temperature')

# ---- LOCAL track ---------------------------------------------------------
print('LOCAL — rasterio point sampling:')
probe(lambda: results.__setitem__('Land Cover',
      local_rasters.sample_points(local_rasters.NLCD_IMG, lons, lats)), 'Land Cover')
probe(lambda: results.__setitem__('Vegetation Mode',
      local_rasters.sample_points(local_rasters.VEGMODE_TIF, lons, lats)), 'Vegetation Mode')
probe(lambda: results.__setitem__('Flammability Index',
      local_rasters.sample_points(local_rasters.FLAMMABILITY_TIF, lons, lats)), 'Flammability Index')

# ---- Daymet SWE/trends: CONDITIONAL on the materialized raster -----------
print('Daymet SWE/trends — materialized raster (conditional):')
daymet_tif = build_daymet_rasters.OUT_TIF
if daymet_tif.exists():
    for band_i, (bname, _, _) in enumerate(build_daymet_rasters.BANDS, start=1):
        label = {'swe_mean': 'Mean Annual SWE', 'swe_trend': 'Trend in SWE',
                 'prcp_trend': 'Trend in precipitation',
                 'tmax_trend': 'Trend in temperature'}[bname]
        probe(lambda p=daymet_tif, i=band_i, l=label: results.__setitem__(
            l, local_rasters.sample_points(p, lons, lats, band=i)), label)
else:
    pending = ['Mean Annual SWE', 'Trend in SWE', 'Trend in precipitation', 'Trend in temperature']
    print(f'  PENDING (soft-skip): {daymet_tif} not built yet — Daymet migration not landed.')
    for p in pending:
        print(f'    - {p}')

# --------------------------------------------------------------------------
# Report + gate.
# --------------------------------------------------------------------------
print('\n' + '=' * 74)
print(f'{"feature":<34}{"finite/N":>10}{"sample value":>22}')
print('=' * 74)
failures = []
sparse_ok = []
for name, arr in results.items():
    finite = np.isfinite(arr)
    nfin = int(finite.sum())
    ex = f'{arr[finite][0]:.4g}' if nfin else '--'
    note = ''
    if name in errored:
        note = '  <- ERROR (probe raised)'
        failures.append(name)
    elif nfin == 0:
        if name in SPARSE:
            note = '  <- sparse: call ok, no coverage at sample (soft-skip)'
            sparse_ok.append(name)
        else:
            failures.append(name)
    print(f'{name:<34}{f"{nfin}/{N}":>10}{ex:>22}{note}')
for name in pending:
    print(f'{name:<34}{"PENDING":>10}{"(migration not landed)":>22}')
print('=' * 74)

if failures:
    print(f'\nFAIL: {len(failures)} feature(s) faulted (probe raised, or unexpected '
          'all-NaN in a non-sparse feature):')
    for f in failures:
        print(f'  - {f}')
    sys.exit(1)
if sparse_ok:
    print(f'\nNote: {len(sparse_ok)} expected-sparse feature(s) all-NaN at this sample '
          f'(probe succeeded, not a fault): {", ".join(sparse_ok)}.')
if pending:
    print(f'\nPASS (with {len(pending)} Daymet feature(s) PENDING — re-run once the '
          'materialized raster exists to gate them).')
else:
    print('\nPASS: every probed feature returned finite values (or is expected-sparse '
          'with a successful call). Wiring looks good.')
