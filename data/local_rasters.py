"""LOCAL-track feature sourcing: sample downloaded source rasters at point
coordinates, replacing four of the lost custom GEE assets with first-party,
account-independent local files.

Why this exists
---------------
Access to the original ``ee-abrupt-thaw`` project was lost (2026-07-10) and its
13 custom uploaded assets have no local copies (see TASKS T0 / memory
``ee-project-access-lost``). Rather than re-upload assets (and re-inherit the
project-scoped-asset fragility), the feature side is rebuilt from first-party
sources with **zero custom uploaded assets**, split into two tracks:

  * GEE track   -> ``gee_features.py`` (inline computation on public catalog data)
  * LOCAL track -> this module (rasterio point-sampling of downloaded rasters)

The LOCAL track samples downloaded source rasters at point/grid coordinates:
ALFRESCO flammability + vegetation mode, NLCD 2016 land cover, and the Daymet V4
SWE + SWE/precip/temp trends. The first three have no GEE-catalog upstream; the
Daymet layer does, but its deep temporal reductions hang when sampled live at
scattered points (T30), so it is materialized once to a local raster by
``build_daymet_rasters.py``. Source rasters live under ``data/`` (git-ignored;
regenerate via ``fetch_alfresco.py``, ``build_daymet_rasters.py``, or — for NLCD
— user-provided) and are documented in ``PIPELINE.md`` -> "Features".

Sampling semantics
------------------
The original assets were sampled at points with ``ee.Reducer.mean()`` at native
scale, which for a single point reduces to the covering pixel. The faithful
Python analogue is **nearest-neighbour** sampling of the covering pixel, which is
also the only correct choice for the categorical layers (land cover, vegetation
mode). We therefore use nearest for every LOCAL feature and document it in the
methods table. Points are reprojected from WGS84 lon/lat into each raster's own
CRS before sampling (CRSs differ: ALFRESCO EPSG:3338, NLCD WGS84-Albers,
Obu EPSG:3995). Nodata and floating sentinels resolve to ``NaN`` so XGBoost's
native missing-value routing applies (matching the points/datacube contract).
"""

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA

# --------------------------------------------------------------------------
# Source raster paths (git-ignored; see module docstring for provenance).
# --------------------------------------------------------------------------
FLAMMABILITY_TIF = (DATA / 'alfresco' /
                    'alfresco_relative_flammability_cru_ts40_historical_1900_1999_iem.tif')
VEGMODE_TIF = (DATA / 'alfresco' /
               'alfresco_vegetation_mode_1950-2008_historical.tif')
NLCD_IMG = DATA / 'NLCD2016' / 'NLCD_2016_Land_Cover_AK_20200724.img'
OBU_TIF = (DATA / 'Obu2019' /
           'UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif')

# Daymet V4 reductions (Mean Annual SWE + SWE/precip/temp trends) materialized to
# one 4-band local raster by ``build_daymet_rasters.py``. These are deep temporal
# reductions that hang when point-sampled live on GEE (T30), so they are computed
# once and read from disk here like the other LOCAL rasters.
DAYMET_TIF = DATA / 'daymet' / 'daymet_v4_reductions_1km_3338.tif'
DAYMET_BANDS = {            # feature name -> 1-indexed band in DAYMET_TIF
    'Mean Annual SWE': 1,
    'Trend in SWE': 2,
    'Trend in precipitation': 3,
    'Trend in temperature': 4,
}

_SENTINEL = -1e30  # ALFRESCO/Obu use large-negative float nodata


def sample_points(path, lons, lats, band: int = 1) -> np.ndarray:
    """Nearest-neighbour sample ``path`` (band ``band``) at WGS84 ``lons``/``lats``.

    Points are reprojected into the raster's CRS. Nodata, non-finite values, and
    large-negative sentinels resolve to ``NaN``. Points outside the raster
    footprint also resolve to ``NaN`` (nodata) or, for rasters with no declared
    nodata (e.g. NLCD, where background is code ``0``), to the edge/background
    value returned by rasterio. Returns a float array aligned to the inputs.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out = np.full(lons.shape, np.nan, dtype=float)
    # Only sample finite, in-range coordinates (datacube off-ROI cells carry a
    # -9999 lon/lat fill that must not reach rasterio's sampler).
    ok = (np.isfinite(lons) & np.isfinite(lats)
          & (np.abs(lons) <= 180) & (np.abs(lats) <= 90))
    if not ok.any():
        return out
    with rasterio.open(path) as ds:
        xs, ys = warp_transform('EPSG:4326', ds.crs,
                                lons[ok].tolist(), lats[ok].tolist())
        vals = np.array(
            [v[0] for v in ds.sample(zip(xs, ys), indexes=[band])],
            dtype=float,
        )
        nodata = ds.nodata
    vals[~np.isfinite(vals)] = np.nan
    if nodata is not None:
        vals[vals == nodata] = np.nan
    vals[vals <= _SENTINEL] = np.nan
    out[ok] = vals
    return out


if __name__ == '__main__':
    # Smoke test: sample every LOCAL raster at a few known Alaska sites.
    sites = {'Fairbanks': (-147.72, 64.84),
             'Utqiagvik': (-156.79, 71.29),
             'Anchorage': (-149.90, 61.22)}
    lons = [v[0] for v in sites.values()]
    lats = [v[1] for v in sites.values()]
    print('Flammability :', dict(zip(sites, np.round(sample_points(FLAMMABILITY_TIF, lons, lats), 4))))
    print('Vegetation   :', dict(zip(sites, sample_points(VEGMODE_TIF, lons, lats))))
    print('Land cover    :', dict(zip(sites, sample_points(NLCD_IMG, lons, lats))))
    print('Obu PerProb  :', dict(zip(sites, np.round(sample_points(OBU_TIF, lons, lats), 3))))
