"""Materialize the four Daymet V4 reductions to a local raster, reproducibly.

Why this exists (TASKS T30)
---------------------------
``Mean Annual SWE`` and the ``Trend in SWE / precipitation / temperature``
features are deep temporal reductions of daily Daymet V4 (~30 years x ~365
daily images, see ``gee_features``). Earth Engine evaluates such graphs lazily
with no persisted intermediate, so point-sampling them at all 19,540 ThawDB
points re-runs the whole reduction per point and hangs (SWE killed >10 min at
1 km; the trends never completed). This is the same shape as the FIRMS max.

The fix is to **compute the reduction once per output tile and write it to a
local raster**, then have the pipeline read that raster cheaply. We pull the
grid with ``ee.data.computePixels`` — the high-volume raster endpoint, sibling
of the ``ee.data.computeFeatures`` call in ``ee_sampling.py`` — one tile at a
time. Each tile computes the reduction once over its own footprint (the bounded
"compute once over the footprint" pattern the datacube's ``sampleRectangle``
already relies on); a tile that exceeds the request/compute limit is split into
quadrants and retried. Tiles are stitched on a single pre-defined pixel grid, so
alignment is exact by construction.

Asset-free contract (settings.py / TASKS T0)
--------------------------------------------
No custom uploaded asset is involved — the reduction is computed on the fly from
public catalog Daymet V4 and streamed straight to disk. (An intermediate GEE
asset was tried first but the compute project ``abrupt-thaw-indicators`` has no
asset home; ``geedim``'s tiled downloader is dependency-incompatible with the
pinned ``earthengine-api``/``rasterio`` stack — hence this direct
``computePixels`` path.) The pipeline's source of truth is the downloaded local
GeoTIFF (git-ignored, under ``data/daymet/``), sampled by BOTH tracks like the
other LOCAL rasters: ``build_feature_table.py`` via ``local_rasters.sample_points``
and ``build_prediction_data.py`` via ``sample_local`` at cell centres. This
honors the ``settings.py`` "no ASSET_ROOT / no runtime custom-asset dependency"
guardrail.

Reproducibility
---------------
The reductions come verbatim from ``gee_features`` (SWE mean + three
``linearFit`` slopes over ``TREND_YEARS`` = 1991-2020). Grid: EPSG:3338 (Alaska
Albers) at Daymet's native 1 km, snapped to a 1 km origin and covering
``data/roi.geojson`` (the datacube domain). Re-running reproduces the same
raster; an existing local file is skipped, so it is safe to re-run.

Band order (1-indexed for rasterio; see BANDS):
    1 = swe_mean    (mm)      Mean Annual SWE
    2 = swe_trend   (mm/yr)   Trend in SWE
    3 = prcp_trend  (mm/yr)   Trend in precipitation
    4 = tmax_trend  (degC/yr) Trend in temperature
"""

import json
import math
import time
from pathlib import Path
import sys

import ee
import geemap
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA, EE_PROJECT
import gee_features

# --------------------------------------------------------------------------
# Derivation parameters (single source of truth for the materialized raster).
# --------------------------------------------------------------------------
CRS = 'EPSG:3338'          # Alaska Albers, matches the other LOCAL rasters
SCALE = 1000               # Daymet V4 native resolution (m)
NODATA = -9999.0           # masked (no-Daymet) pixels; local_rasters maps -> NaN
TILE = 256                 # initial tile size (px); subdivided on size/memory error
OUT_TIF = DATA / 'daymet' / 'daymet_v4_reductions_1km_3338.tif'
ROI_GEOJSON = DATA / 'roi.geojson'

# (band name, gee_features constructor, source band it emits) in output order.
BANDS = [
    ('swe_mean',   gee_features.mean_annual_swe, 'swe'),
    ('swe_trend',  gee_features.swe_trend,       'scale'),
    ('prcp_trend', gee_features.precip_trend,    'scale'),
    ('tmax_trend', gee_features.temp_trend,      'scale'),
]

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0
_SIZE = ('must be less than', 'request size', 'too large', 'user memory limit',
         'memory limit exceeded', 'computed value is too large')
_TRANSIENT = ('timed out', 'timeout', 'deadline', 'try again', 'temporarily',
              'backend error', 'internal error', 'service unavailable',
              'rate limit', 'too many requests', 'quota', '429', '500', '502', '503')


def reductions_image() -> ee.Image:
    """The four Daymet reductions as one multiband image, bands named per BANDS.
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
    """Pixel grid (x0, y0, W, H) in CRS units: 1 km cells snapped to a 1 km
    origin, covering the ROI's bounding box."""
    ring = roi.bounds(maxError=1, proj=ee.Projection(CRS)).coordinates().getInfo()[0]
    xs = [c[0] for c in ring]
    ys = [c[1] for c in ring]
    x0 = math.floor(min(xs) / SCALE) * SCALE
    y0 = math.ceil(max(ys) / SCALE) * SCALE
    w = int(math.ceil((max(xs) - x0) / SCALE))
    h = int(math.ceil((y0 - min(ys)) / SCALE))
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
    print(f'\nDaymet reductions materialized: {OUT_TIF}')
