"""Shared-computation point sampling for deep-reduction GEE features.

Background (TASKS T30)
----------------------
``build_feature_table.py`` samples most features with ``add_feature`` — it maps
``reduceRegion`` over a FeatureCollection of points and trusts that
``computeFeatures`` returns rows in input order. That is fine for flat catalog
rasters (cheap random access), but degrades catastrophically for an image built
on a deep temporal reduction (e.g. a multi-hundred/thousand-image temporal max or
trend): mapping the reduction over 19,540 scattered points re-evaluates it at
every point with no tile sharing, and hangs (>26 min at full N, killed). Such
deep reductions (Daymet SWE/trends, MODIS MCD64A1 fire history) are now instead
materialized once to LOCAL rasters (build_daymet_rasters.py /
build_modis_fire_rasters.py); this module remains for any that still need live
GEE point sampling.

This module is the fallback used ONLY for features that cannot be loaded the old
way (see TASKS T30). The core idea is the one the datacube path already proves
fast: **compute the reduction once, then read every point from it.** A single
``image.reduceRegions`` over all points streams the collection against tiles that
are computed once over the points' footprint — the same reason the datacube's
``sampleRectangle`` completes in minutes, not hours.

Design notes (both learned empirically, see the T30 parity smoke):

* **One call, not scattered chunks.** ``reduceRegions`` computes the image over
  the tiles covering the *whole footprint* of the input collection. The Thaw DB
  points are not spatially sorted, so any index-based chunk spans the whole state
  and would re-trigger the full statewide computation per chunk (~20x redundant).
  So we issue a single ``reduceRegions`` and retrieve it via
  ``ee.data.computeFeatures`` (which paginates — ``getInfo`` caps at ~5,000
  features, the only reason chunking was ever considered).
* **Key by row id, never by position.** Each point carries an explicit ``row_id``
  and results are reassembled by that key. ``reduceRegions`` does not preserve
  collection order, and mis-ordering would silently scramble the training table.
* Transient errors are retried; memory errors escalate ``tileScale``.

Value semantics match the ``reduceRegion(reducer, scale)`` definition used for
every other point feature: a Point geometry reduces the single pixel it falls in
at ``scale``. This is *methodological* parity with the datacube path, not numeric
equality — the feature table trains the model and the datacube serves it, on
deliberately different grids.
"""

import time

import ee
import numpy as np
import pandas as pd

MAX_RETRIES = 3            # transient-error retries
RETRY_BASE_DELAY = 5.0     # seconds; backoff is RETRY_BASE_DELAY * attempt
MAX_TILE_SCALE = 16        # ceiling for memory-driven tileScale escalation

_TRANSIENT = (
    'timed out', 'timeout', 'deadline', 'try again', 'temporarily',
    'backend error', 'internal error', 'service unavailable',
    'rate limit', 'too many requests', 'quota',
    '429', '500', '502', '503',
)
_MEMORY = (
    'user memory limit', 'memory limit exceeded', 'too large',
    'computed value is too large',
)


def _classify(err: Exception) -> str:
    """Bucket an Earth Engine error as 'memory', 'transient', or 'fatal'."""
    msg = str(err).lower()
    if any(m in msg for m in _MEMORY):
        return 'memory'
    if any(m in msg for m in _TRANSIENT):
        return 'transient'
    return 'fatal'


def _compute_reduced(img: ee.Image, points_fc: ee.FeatureCollection,
                     reducer: ee.Reducer, scale: float, crs: str) -> pd.DataFrame:
    """Run one ``reduceRegions`` and pull it as a DataFrame, with retry +
    tileScale escalation. Raises on a fatal error or after exhausting transient
    retries — the caller decides whether to record the failure and continue, so
    a genuine failure is never swallowed silently.
    """
    tile_scale = 1
    attempt = 0
    while True:
        try:
            reduced = img.reduceRegions(
                collection=points_fc, reducer=reducer, scale=scale,
                crs=crs, tileScale=tile_scale,
            )
            return ee.data.computeFeatures({
                'expression': reduced,
                'fileFormat': 'PANDAS_DATAFRAME',
            })
        except ee.EEException as e:
            kind = _classify(e)
            if kind == 'memory' and tile_scale < MAX_TILE_SCALE:
                tile_scale *= 2
                continue
            if kind == 'transient' and attempt < MAX_RETRIES:
                attempt += 1
                time.sleep(RETRY_BASE_DELAY * attempt)
                continue
            raise


def sample_points_reduceregions(lons, lats, image: ee.Image, reducer: ee.Reducer,
                                scale: float, band: str,
                                crs: str = 'EPSG:4326') -> np.ndarray:
    """Reduce ``image``'s ``band`` at each ``(lon, lat)`` via a single, shared
    ``reduceRegions`` call.

    Returns a float array aligned to the inputs; points that are masked or fall
    outside the image footprint are ``NaN``. Raises if the reduction fails fatally.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    n = lons.size
    out = np.full(n, np.nan, dtype=float)

    points_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(lons[i]), float(lats[i])]),
                   {'row_id': i})
        for i in range(n)
    ])
    df = _compute_reduced(image.select(band), points_fc, reducer, scale, crs)

    if 'row_id' not in df.columns:
        return out
    # The reduced value column is the numeric one that is not the key. (Masked
    # points simply have NaN there; if every point was masked the column may be
    # absent entirely, leaving the all-NaN default.)
    value_cols = [c for c in df.columns
                  if c != 'row_id' and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        return out
    value_col = band if band in value_cols else value_cols[0]

    rid = df['row_id'].to_numpy().astype(int)
    val = df[value_col].to_numpy(dtype=float)
    out[rid] = val
    return out


def sample_points_reduceregions_chunked(lons, lats, image: ee.Image,
                                        reducer: ee.Reducer, scale: float, band: str,
                                        crs: str = 'EPSG:4326',
                                        chunk: int = 20000) -> np.ndarray:
    """Chunked ``reduceRegions`` for a CHEAP raster sampled at very many points.

    Used to serve native-scale terrain at the datacube's ~1e6 grid cell centres
    (TASKS T37): a 1 km reproject would pyramid-aggregate 10 m slope/aspect
    (probe_native_serve), so the datacube must read the native pixel at each cell
    centre — the same construction the point path uses per training point. A
    single ``reduceRegions`` over ~1e6 client-built points is too large a request,
    so points are split into contiguous chunks. Because the datacube grid is
    raster-ordered, consecutive chunks are spatially compact strips, so each
    ``reduceRegions`` still covers only a bounded footprint.

    NOT for deep temporal reductions — for those a single call is mandatory (see
    the module docstring); chunking one re-triggers the whole reduction per chunk.
    Invalid/off-grid coords (``|lon|>180`` or ``|lat|>90``, e.g. the -9999 off-ROI
    fill) sample to ``NaN``, matching ``local_rasters.sample_points``.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out = np.full(lons.shape, np.nan, dtype=float)
    ok = (np.isfinite(lons) & np.isfinite(lats)
          & (np.abs(lons) <= 180) & (np.abs(lats) <= 90))
    idx = np.flatnonzero(ok)
    for start in range(0, idx.size, chunk):
        sl = idx[start:start + chunk]
        out[sl] = sample_points_reduceregions(
            lons[sl], lats[sl], image, reducer, scale, band, crs)
    return out


def add_feature_reduceregions(df, lons, lats, image: ee.Image, reducer: ee.Reducer,
                              scale: float, name: str, band: str,
                              crs: str = 'EPSG:4326') -> None:
    """``add_feature`` analogue for deep-reduction images: assign the shared,
    key-aligned ``reduceRegions`` result to ``df[name]``. Argument order mirrors
    ``add_feature`` (minus the FeatureCollection, plus ``lons``/``lats``) so a
    feature can be flipped from the old path to this one with a one-line change.
    """
    df[name] = sample_points_reduceregions(lons, lats, image, reducer, scale, band, crs)
