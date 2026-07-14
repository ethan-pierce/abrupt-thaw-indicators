"""Materialize the MODIS MCD64A1 fire-history reductions to a local raster (T36).

Why this exists
---------------
``Time Since Last Fire`` and ``Burn Count`` are deep temporal reductions of the
MODIS MCD64A1 monthly burned-area product (~280 monthly images over
``FIRE_RECORD``, see ``gee_features``). Earth Engine evaluates such graphs lazily
with no persisted intermediate, so point-sampling them at all ~19,540 ThawDB
points re-runs the whole reduction per point and hangs — the same shape as the
Daymet reductions (T30) and the retired FIRMS max.

The fix, identical to ``build_daymet_rasters.py``, is to **compute the reduction
once per output tile and write it to a local raster**, then have the pipeline
read that raster cheaply. Tiles are pulled with ``ee.data.computePixels`` (the
high-volume raster endpoint) one at a time onto a single pre-defined pixel grid,
so alignment is exact by construction; a tile that exceeds the request/compute
limit is split into quadrants and retried.

Asset-free contract (settings.py / TASKS T0)
--------------------------------------------
No custom uploaded asset is involved — the reduction is computed on the fly from
the public catalog ``MODIS/061/MCD64A1`` and streamed straight to disk. The
pipeline's source of truth is the downloaded local GeoTIFF (git-ignored, under
``data/modis_fire/``), sampled by BOTH tracks like the other LOCAL rasters:
``build_feature_table.py`` via ``local_rasters.sample_points`` and
``build_prediction_data.py`` via ``sample_local`` at cell centres.

Resolution
----------
Kept near MCD64A1's ~500 m native scale (``SCALE``), finer than the 1 km serve
grid — the datacube resamples it to 1 km downstream (like the other sub-km LOCAL
rasters). Grid: EPSG:3338 (Alaska Albers), snapped to a ``SCALE`` origin and
covering union(datacube ROI, ThawDB points) + ``MARGIN`` (some training points
lie south of the interior/Arctic ROI). Re-running reproduces the same raster; an
existing local file is skipped, so it is safe to re-run.

Right-censoring
---------------
Both bands are right-censored to ``FIRE_RECORD`` (see ``gee_features``): "no fire
since 2001" != "never burned". ``Time Since Last Fire`` is capped at the record
length for never-burned pixels; ``Burn Count`` counts burns within the record
only.

Band order (1-indexed for rasterio; see BANDS):
    1 = tslf        (yr)   Time Since Last Fire (capped at record length)
    2 = burn_count  (count) Burn Count over the record
"""

import json
import math
import time
from pathlib import Path
import sys

import ee
import geemap
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA, EE_PROJECT
import gee_features

# --------------------------------------------------------------------------
# Derivation parameters (single source of truth for the materialized raster).
# --------------------------------------------------------------------------
CRS = 'EPSG:3338'          # Alaska Albers, matches the other LOCAL rasters
SCALE = 500                # ~MCD64A1 native resolution (m); datacube resamples to 1 km
NODATA = -9999.0           # off-coverage pixels; local_rasters maps -> NaN
TILE = 256                 # initial tile size (px); subdivided on size/memory error
MARGIN = 5000              # m of padding around the covered extent (edge points)
OUT_TIF = DATA / 'modis_fire' / 'mcd64a1_fire_history_500m_3338.tif'
ROI_GEOJSON = DATA / 'roi.geojson'
# The feature table samples this raster at every ThawDB point, so the grid must
# cover the training points, not just the datacube ROI (some points lie south of
# the interior/Arctic ROI). Extent = union(ROI bbox, ThawDB point bbox) + MARGIN.
THAWDB_CSV = DATA / 'Alaska_Permafrost_Thaw_Database_v2.0.0.csv'

# (band name, gee_features constructor, source band it emits) in output order.
BANDS = [
    ('tslf',       gee_features.time_since_last_fire, 'tslf'),
    ('burn_count', gee_features.burn_count,           'burn_count'),
]

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0
_SIZE = ('must be less than', 'request size', 'too large', 'user memory limit',
         'memory limit exceeded', 'computed value is too large')
_TRANSIENT = ('timed out', 'timeout', 'deadline', 'try again', 'temporarily',
              'backend error', 'internal error', 'service unavailable',
              'rate limit', 'too many requests', 'quota', '429', '500', '502', '503')


def reductions_image() -> ee.Image:
    """The two MCD64A1 reductions as one multiband image, bands named per BANDS.
    Masked pixels are filled with NODATA so computePixels returns them explicitly."""
    bands = [ctor().select(src).rename(name) for name, ctor, src in BANDS]
    return ee.Image.cat(bands).unmask(NODATA)


def _roi() -> ee.Geometry:
    with open(ROI_GEOJSON) as f:
        return geemap.geojson_to_ee(json.load(f)).geometry()


def _classify(err: Exception) -> str:
    msg = str(err).lower()
    if any(m in msg for m in _SIZE):
        return 'size'
    if any(m in msg for m in _TRANSIENT):
        return 'transient'
    return 'fatal'


def _grid(roi: ee.Geometry):
    """Pixel grid (x0, y0, W, H) in CRS units: SCALE-m cells snapped to a SCALE
    origin, covering union(datacube ROI, ThawDB points) + MARGIN."""
    ring = roi.bounds(maxError=1, proj=ee.Projection(CRS)).coordinates().getInfo()[0]
    xs = [c[0] for c in ring]
    ys = [c[1] for c in ring]
    # Fold in the ThawDB points (projected to CRS) so no training point is off-grid.
    pts = pd.read_csv(THAWDB_CSV, sep=',', encoding='latin1')
    px, py = warp_transform('EPSG:4326', CRS,
                            pts['Longitude'].tolist(), pts['Latitude'].tolist())
    xmin, xmax = min(min(xs), *px) - MARGIN, max(max(xs), *px) + MARGIN
    ymin, ymax = min(min(ys), *py) - MARGIN, max(max(ys), *py) + MARGIN
    x0 = math.floor(xmin / SCALE) * SCALE
    y0 = math.ceil(ymax / SCALE) * SCALE
    w = int(math.ceil((xmax - x0) / SCALE))
    h = int(math.ceil((y0 - ymin) / SCALE))
    return x0, y0, w, h


def download(roi: ee.Geometry) -> Path:
    """Fetch the reduction tile-by-tile onto a single grid and write OUT_TIF.
    No-op if it already exists."""
    OUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    if OUT_TIF.exists():
        print(f'skip download (exists): {OUT_TIF}')
        return OUT_TIF

    x0, y0, W, H = _grid(roi)
    print(f'grid: {W}x{H} px @ {SCALE} m, origin ({x0}, {y0}) {CRS}')
    image = reductions_image()
    names = [name for name, _, _ in BANDS]
    out = np.full((len(BANDS), H, W), NODATA, dtype=np.float32)

    def fetch(col, row, w, h, attempt=0):
        req = {
            'expression': image,
            'fileFormat': 'NUMPY_NDARRAY',
            'grid': {
                'dimensions': {'width': w, 'height': h},
                'affineTransform': {
                    'scaleX': SCALE, 'shearX': 0, 'translateX': x0 + col * SCALE,
                    'shearY': 0, 'scaleY': -SCALE, 'translateY': y0 - row * SCALE,
                },
                'crsCode': CRS,
            },
        }
        try:
            arr = ee.data.computePixels(req)
        except ee.EEException as e:
            kind = _classify(e)
            if kind == 'size' and (w > 1 or h > 1):
                w1, h1 = (w + 1) // 2, (h + 1) // 2
                for c, r, ww, hh in ((col, row, w1, h1), (col + w1, row, w - w1, h1),
                                     (col, row + h1, w1, h - h1),
                                     (col + w1, row + h1, w - w1, h - h1)):
                    if ww > 0 and hh > 0:
                        fetch(c, r, ww, hh)
                return
            if kind == 'transient' and attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                fetch(col, row, w, h, attempt + 1)
                return
            raise
        for i, name in enumerate(names):
            out[i, row:row + h, col:col + w] = arr[name]

    ntiles = math.ceil(W / TILE) * math.ceil(H / TILE)
    done = 0
    for row in range(0, H, TILE):
        for col in range(0, W, TILE):
            fetch(col, row, min(TILE, W - col), min(TILE, H - row))
            done += 1
            print(f'  tile {done}/{ntiles}')

    transform = from_origin(x0, y0, SCALE, SCALE)
    with rasterio.open(
        OUT_TIF, 'w', driver='GTiff', height=H, width=W, count=len(BANDS),
        dtype='float32', crs=CRS, transform=transform, nodata=NODATA,
        compress='deflate',
    ) as dst:
        for i, name in enumerate(names, start=1):
            dst.write(out[i - 1], i)
            dst.set_band_description(i, name)
    print(f'wrote {OUT_TIF}')
    return OUT_TIF


if __name__ == '__main__':
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
    download(_roi())
    print(f'\nMODIS MCD64A1 fire history materialized: {OUT_TIF}')
