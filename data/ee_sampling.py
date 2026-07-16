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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    # Concurrency-limit errors from the tiled fan-out (T47): EE emits these under
    # parallel reduceRegions load; they are transient (back off + retry), not fatal.
    'too many concurrent', 'concurrent aggregations', 'concurrent requests',
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


def sample_native_multiband_tiled(
    lon2d, lat2d, image: ee.Image, bands, scale: float,
    reducer: ee.Reducer = None, crs: str = 'EPSG:4326',
    tile: int = 128, workers: int = 8, log=None,
) -> dict:
    """Point-sample a MULTI-BAND image at every grid-cell centre via 2-D tiled,
    concurrent ``reduceRegions`` — the datacube's native-scale server (TASKS T47).

    ``lon2d``/``lat2d`` are the 2-D ``(row, col)`` cell-centre coordinates of the
    prediction grid in GEE-native (unflipped) orientation. The grid is cut into
    square ``tile``x``tile`` INDEX blocks; each block is spatially compact, so one
    ``reduceRegions`` over its points touches only a bounded footprint of the
    native mosaic. This replaces ``sample_points_reduceregions_chunked``, whose
    index chunks of 20k row-major points spanned Alaska's full ~29 deg E-W width,
    re-triggering a statewide computation per chunk (~1.7 min/chunk) — 0 features
    in 3h14m at statewide scale.

    All ``bands`` must share ``scale``: they are reduced in ONE pass with one
    ``reducer`` (``mean`` by default). Merging bands is numerically identical to
    sampling each alone — ``reduceRegions`` reduces every band independently — so
    train/serve parity (a Point reduces the single native pixel it falls in at
    ``scale``) is preserved (verified per band by the T47 one-tile parity gate).

    Results are reassembled by band **name** and ``row_id`` (never position;
    ``reduceRegions`` does not preserve order). Per-band masking yields ragged NaN
    across bands, and off-grid cells (``|lon|>180`` / ``|lat|>90`` / non-finite,
    e.g. the -9999 off-ROI fill) stay NaN — matching ``local_rasters.sample_points``.
    A band masked over a whole tile is simply absent from that tile's frame and
    stays NaN. Tiles with no valid cell are skipped entirely (the big statewide
    win: Alaska is diagonal, so many bbox tiles are all off-ROI).

    Returns ``{band: 2-D array}`` in native (unflipped) orientation; the caller
    flips to match the rest of the stack. ``log(done, total)`` is called after
    each tile completes if provided.
    """
    reducer = ee.Reducer.mean() if reducer is None else reducer
    lon2d = np.asarray(lon2d, dtype=float)
    lat2d = np.asarray(lat2d, dtype=float)
    nrows, ncols = lon2d.shape
    out = {b: np.full((nrows, ncols), np.nan, dtype=float) for b in bands}
    img = image.select(bands)

    tiles = [(r0, min(r0 + tile, nrows), c0, min(c0 + tile, ncols))
             for r0 in range(0, nrows, tile)
             for c0 in range(0, ncols, tile)]

    def _do_tile(t):
        """Worker: build the tile's valid-point FC and run one reduceRegions.
        Returns ``(t, oki, df)`` or ``None`` for an all-off-grid tile. ``row_id``
        is the LOCAL index into this tile's valid points."""
        r0, r1, c0, c1 = t
        lo = lon2d[r0:r1, c0:c1].ravel()
        la = lat2d[r0:r1, c0:c1].ravel()
        ok = (np.isfinite(lo) & np.isfinite(la)
              & (np.abs(lo) <= 180) & (np.abs(la) <= 90))
        oki = np.flatnonzero(ok)
        if oki.size == 0:
            return None
        points_fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(lo[i]), float(la[i])]),
                       {'row_id': int(j)})
            for j, i in enumerate(oki)
        ])
        df = _compute_reduced(img, points_fc, reducer, scale, crs)
        return t, oki, df

    def _scatter(res):
        """Main-thread reassembly (no shared-array races): place each band's
        values back into its (row, col) block by row_id, valid cells only."""
        t, oki, df = res
        r0, r1, c0, c1 = t
        h, w = r1 - r0, c1 - c0
        if 'row_id' not in df.columns:
            return
        rid = df['row_id'].to_numpy().astype(int)
        # reduceRegions names each MULTIband mean-output column by its band, but a
        # SINGLE-band image's lone output column is named 'mean' (not the band). So
        # a single-band group would find `b not in df.columns` and silently return
        # all-NaN. Mirror sample_points_reduceregions' fallback: for a one-band call
        # only, take the sole numeric non-key column. For a true multiband call a
        # missing band means it was masked over the whole tile -> stays NaN (we must
        # NOT borrow another band's column), so the fallback is gated on len==1.
        numeric = [c for c in df.columns
                   if c != 'row_id' and pd.api.types.is_numeric_dtype(df[c])]
        for b in bands:
            if b in df.columns:
                col = b
            elif len(bands) == 1 and len(numeric) == 1:
                col = numeric[0]
            else:
                continue  # band masked over the whole tile -> stays NaN
            valid_vals = np.full(oki.size, np.nan, dtype=float)
            valid_vals[rid] = df[col].to_numpy(dtype=float)
            block = np.full(h * w, np.nan, dtype=float)
            block[oki] = valid_vals
            out[b][r0:r1, c0:c1] = block.reshape(h, w)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_do_tile, t) for t in tiles]
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                _scatter(res)
            done += 1
            if log is not None:
                log(done, len(tiles))
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
