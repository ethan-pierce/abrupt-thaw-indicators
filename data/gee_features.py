"""GEE-track feature sourcing: inline computation from public Google Earth
Engine catalog datasets, replacing the climate/terrain custom assets that were
lost with the ``ee-abrupt-thaw`` project (see TASKS T0 / memory
``ee-project-access-lost``).

Every constructor here returns an ``ee.Image`` built only from **account-
independent public catalog** datasets (USGS 3DEP, NASA/ORNL Daymet V4, NASA
FIRMS) — there are **no custom uploaded assets and no ``ASSET_ROOT``
dependency**. Both ``build_feature_table.py`` (point sampling) and
``build_prediction_data.py`` (gridded datacube) import these so the two paths
share one definition and cannot drift.

Reconstruction note
-------------------
The original assets' exact build parameters were lost with the project, so the
choices below are **documented reconstructions**, not a byte-match. This is
acceptable for the ThawDB v2.0.0 rebuild (see PIPELINE.md). Record these in the
methods table (TASKS T29 remainder):

  * Curvature   : Zevenbergen-Thorne (1987) mean curvature of USGS/3DEP/10m,
                  computed in EPSG:3338 at an analysis cell size of window/2 so a
                  3x3 window spans the named smoothing window (500 m or 2 km).
  * Daymet trends/climatology window : ``TREND_YEARS`` (30-year normal).
  * FIRMS       : temporal maximum of the MODIS ~4 um brightness temperature
                  (band ``T21``, Kelvin) over the full record.
"""

import ee

# --------------------------------------------------------------------------
# Reconstructed parameters (documented; original build params lost). Keep these
# in sync with the methods table.
# --------------------------------------------------------------------------
TREND_YEARS = (1991, 2020)        # inclusive Daymet window for climatology + trends
CURVATURE_CRS = 'EPSG:3338'       # metric projection so 3x3 spacing == cell size
_DAYMET = 'NASA/ORNL/DAYMET_V4'
_DEM_ID = 'USGS/3DEP/10m'


# ==========================================================================
# Terrain: mean curvature (formerly AK-curvature-500m / AK-curvature-2k)
# ==========================================================================
def mean_curvature(window_m: float) -> ee.Image:
    """Reconstructed TAGEE-family **mean curvature** of the 3DEP 10 m DEM.

    ``window_m`` (500 or 2000) is the DEM-smoothing scale that distinguished the
    two original curvature assets. Method (documented reconstruction): resample
    the DEM to a cell size ``d = window_m / 2`` in EPSG:3338 (so a 3x3
    neighbourhood spans ``window_m``), then form the Zevenbergen-Thorne partial
    derivatives with fixed convolution kernels and evaluate

        H = -[(1 + q^2) r - 2 p q s + (1 + p^2) t] / [2 (1 + p^2 + q^2)^{3/2}]

    where p, q are first and r, s, t second partials. Output band
    ``'MeanCurvature'`` (units 1/m), on the EPSG:3338 grid at cell size ``d``.

    Coarsening uses ``resample('bilinear')`` rather than ``reduceResolution``:
    reduceResolution reads *every* 10 m pixel under each ``d`` cell, which is fine
    for point sampling (tiny neighbourhoods) but makes a statewide
    ``sampleRectangle`` in build_prediction_data.py exceed Earth Engine's 2^31
    pixels-per-request limit. Bilinear samples ~4 source pixels per node, so the
    identical definition works in both the point and datacube paths (train/serve
    parity). The trade-off is no sub-cell averaging; acceptable for the v2.0.0
    reconstruction (no byte-match to the lost assets is required).
    """
    d = float(window_m) / 2.0
    dem = ee.Image(_DEM_ID).select('elevation')
    # Coarsen 10 m -> d m in the metric projection so convolution neighbours are
    # exactly d metres apart and the kernel spacing terms hold. Bounded read.
    dem_d = (dem
             .resample('bilinear')
             .reproject(crs=CURVATURE_CRS, scale=d))

    # Zevenbergen-Thorne (1987) finite-difference kernels on a 3x3 window; rows
    # run north->south, columns west->east; weights fold in the 1/spacing terms.
    p = dem_d.convolve(ee.Kernel.fixed(3, 3, [
        [0, 0, 0], [-1 / (2 * d), 0, 1 / (2 * d)], [0, 0, 0]]))          # dz/dx
    q = dem_d.convolve(ee.Kernel.fixed(3, 3, [
        [0, 1 / (2 * d), 0], [0, 0, 0], [0, -1 / (2 * d), 0]]))          # dz/dy
    r = dem_d.convolve(ee.Kernel.fixed(3, 3, [
        [0, 0, 0], [1 / d**2, -2 / d**2, 1 / d**2], [0, 0, 0]]))         # d2z/dx2
    t = dem_d.convolve(ee.Kernel.fixed(3, 3, [
        [0, 1 / d**2, 0], [0, -2 / d**2, 0], [0, 1 / d**2, 0]]))         # d2z/dy2
    s = dem_d.convolve(ee.Kernel.fixed(3, 3, [
        [-1 / (4 * d * d), 0, 1 / (4 * d * d)], [0, 0, 0],
        [1 / (4 * d * d), 0, -1 / (4 * d * d)]]))                        # d2z/dxdy

    h = dem_d.expression(
        '-((1 + q*q)*r - 2*p*q*s + (1 + p*p)*t) / (2 * pow(1 + p*p + q*q, 1.5))',
        {'p': p, 'q': q, 'r': r, 's': s, 't': t})
    return h.rename('MeanCurvature').reproject(crs=CURVATURE_CRS, scale=d)


# ==========================================================================
# Daymet V4 climate: mean annual SWE + SWE/precip/temp trends
# ==========================================================================
def _annual_daymet(band: str, reducer: ee.Reducer) -> ee.ImageCollection:
    """One image per year in ``TREND_YEARS`` of ``band`` reduced over that year
    by ``reducer``, carrying a constant ``year`` band as the trend's independent
    variable (ordered so linearFit yields [scale, offset])."""
    y0, y1 = TREND_YEARS
    years = ee.List.sequence(y0, y1)

    def per_year(y):
        y = ee.Number(y)
        annual = (ee.ImageCollection(_DAYMET)
                  .filter(ee.Filter.calendarRange(y, y, 'year'))
                  .select(band)
                  .reduce(reducer)
                  .rename(band))
        return (ee.Image.constant(y).float().rename('year')
                .addBands(annual)
                .set('year', y))

    return ee.ImageCollection(years.map(per_year))


def mean_annual_swe() -> ee.Image:
    """Mean annual snow-water-equivalent (mm) over ``TREND_YEARS``: per-year mean
    of daily ``swe``, then the temporal mean across years. Band ``'swe'``."""
    return _annual_daymet('swe', ee.Reducer.mean()).select('swe').mean().rename('swe')


def _linear_trend(band: str, reducer: ee.Reducer) -> ee.Image:
    """Per-pixel OLS slope of the annual series (``year`` -> annual value) via
    ``ee.Reducer.linearFit()``. Returns the ``'scale'`` (slope) band only."""
    return (_annual_daymet(band, reducer)
            .select(['year', band])
            .reduce(ee.Reducer.linearFit())
            .select('scale'))


def swe_trend() -> ee.Image:
    """Trend in annual-mean SWE (mm/yr). Band ``'scale'``."""
    return _linear_trend('swe', ee.Reducer.mean())


def precip_trend() -> ee.Image:
    """Trend in annual-total precipitation (mm/yr). Band ``'scale'``."""
    return _linear_trend('prcp', ee.Reducer.sum())


def temp_trend() -> ee.Image:
    """Trend in annual-mean daily max temperature (deg C/yr). Band ``'scale'``."""
    return _linear_trend('tmax', ee.Reducer.mean())


# ==========================================================================
# MERIT Hydro: hydrological terrain (T34)
# ==========================================================================
# Official catalog asset MERIT/Hydro/v1_0_1 (Yamazaki et al. 2019) — NOT the
# `sat-io` community mirror. Native resolution 3 arc-seconds (~92.77 m in
# EPSG:4326), finer than the 1 km serve grid, which shapes how each band is
# served (see the two constructors below and TASKS T34).
_MERIT_HYDRO = 'MERIT/Hydro/v1_0_1'
MERIT_SCALE = 90                  # round native scale (~92.77 m) both build paths
                                  # sample at, so train/serve agree by construction


def height_above_drainage() -> ee.Image:
    """MERIT Hydro **height above nearest drainage** (metres), raw. Band ``'hnd'``.

    A fine-scale hydrological-terrain control: low ``hnd`` marks valley bottoms
    and drainage lines where water — and thermokarst — concentrate. Served
    **natively** in both paths (like the 3DEP terrain, T37): it is a stored
    height, so a 1 km reproject would average the ~120 native pixels under each
    cell and blur the valley/slope contrast that matters for a fine-scale process.
    Instead both paths point-sample at ``MERIT_SCALE``, so they agree by
    construction (the point path's ``reduceRegion`` and the datacube's per-cell
    ``reduceRegions`` both read the native pixel).
    """
    return ee.Image(_MERIT_HYDRO).select('hnd')


def log_upstream_area() -> ee.Image:
    """Natural log of MERIT Hydro **upstream drainage area** (``upa``, km^2) — the
    water-convergence signal. Band ``'log_upa'``.

    ``upa`` is strictly positive (minimum = one native cell, ~0.0037 km^2) and
    heavy-tailed across many orders of magnitude, so the log is applied **here**,
    before any aggregation. The order matters for the datacube (T34 / T35 bucket
    2): MERIT's ~90 m native grid is finer than the 1 km serve grid, so — unlike
    the natively-served terrain — ``log(upa)`` IS reproject-averaged to 1 km, and
    that average must act on the **log**. A plain ``reproject`` of this image would
    silently average raw ``upa`` first and then log it (``log(mean(upa))``,
    dominated by the few largest channels — empirically verified), so the datacube
    aggregates it with an explicit ``reduceResolution(mean)`` on the native-pinned
    log; the point path samples this same image at native scale (no aggregation),
    so both paths share one definition.
    """
    return ee.Image(_MERIT_HYDRO).select('upa').log().rename('log_upa')


# ==========================================================================
# FIRMS: maximum fire temperature
# ==========================================================================
def max_fire_temp() -> ee.Image:
    """Temporal maximum of the FIRMS MODIS ~4 um brightness temperature
    (band ``T21``, Kelvin) over the full record. Masked where no fire was ever
    detected — that mask also drives the ``Fire Detected`` indicator (A1).
    Band ``'T21'``."""
    return (ee.ImageCollection('FIRMS').select('T21').max().rename('T21'))
