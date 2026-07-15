"""GEE-track feature sourcing: inline computation from public Google Earth
Engine catalog datasets, replacing the climate/terrain custom assets that were
lost with the ``ee-abrupt-thaw`` project (see TASKS T0 / memory
``ee-project-access-lost``).

Every constructor here returns an ``ee.Image`` built only from **account-
independent public catalog** datasets (USGS 3DEP, NASA/ORNL Daymet V4, MERIT
Hydro, NASA/USGS MODIS MCD64A1) — there are **no custom uploaded assets and no
``ASSET_ROOT`` dependency**. Both ``build_feature_table.py`` (point sampling) and
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
  * MCD64A1     : fire history over ``FIRE_RECORD`` — time-since-last-fire +
                  burn-count from the MODIS ~500 m monthly burned-area product
                  (right-censored to the record; see the constructors below).
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


def upstream_area() -> ee.Image:
    """MERIT Hydro **upstream drainage area** (``upa``, km^2), raw — the
    water-convergence signal. Band ``'upa'``.

    ``upa`` is strictly positive (minimum = one native cell, ~0.0037 km^2) and
    heavy-tailed across many orders of magnitude. It is nonetheless served **raw
    and natively** in both paths (like ``hnd`` and the 3DEP terrain, T37): MERIT's
    ~90 m native grid is finer than the 1 km serve grid, so rather than
    reproject-average (a non-tree op that would need the mean taken on the log to
    avoid ``log(mean(upa))`` bias — T35), both paths point-sample the native pixel
    at ``MERIT_SCALE``, so they agree by construction and no averaging occurs.

    No log is applied here (T35): a monotonic transform is a no-op for the XGBoost
    fit, and the canonical feature set is kept raw/physical. The live linear
    baseline (T13) logs this feature within its own preprocessing scope, where a
    heavy tail actually matters.
    """
    return ee.Image(_MERIT_HYDRO).select('upa')


# ==========================================================================
# MODIS MCD64A1 burned area: fire history (T36)
# ==========================================================================
# Combined Terra+Aqua monthly burned-area product (Giglio et al. 2018), ~500 m
# native. Replaces the FIRMS max-fire-temperature / Fire-Detected pair (T4/T18,
# reverted): FIRMS ``T21`` is the peak ~4 um brightness of a single detection —
# an instantaneous intensity, not a fire *regime* — and the binary "ever
# detected" indicator is near-constant at 1 km. MCD64A1 gives a fire *history*:
# how recently, and how often, a location has burned within the record.
#
# Two features are derived, both **deep temporal reductions** over the ~280
# monthly images. Like the Daymet reductions (T30) they hang if point-sampled
# live on GEE, so they are materialized once to a LOCAL ~500 m raster by
# ``build_modis_fire_rasters.py`` and read from disk by both tracks. Kept near
# the native ~500 m; the datacube resamples to 1 km downstream.
#
# 24-YEAR RIGHT-CENSORING (document in the methods table): the record spans only
# ``FIRE_RECORD`` (~24 yr from 2001), so "no fire detected since 2001" is NOT
# "never burned" — a pixel last burned in 1998 is indistinguishable from one that
# last burned centuries ago. Both features are therefore right-censored: Time
# Since Last Fire is capped at the record length, and Burn Count counts burns
# within the record only, not the full fire history.
#
# HIGH-LATITUDE COVERAGE GAP (document alongside): MODIS burned-area detection
# thins out on the far-north Arctic coast — above ~70 deg N the MCD64A1 QA domain
# drops out (empirically ~11% of ThawDB points, all 70.0-71.2 deg N, verified
# T36). Those pixels return NaN for both features (honest "no MODIS fire
# evidence", routed natively by XGBoost), NOT a censored value. T39's dry-run
# reports the statewide NaN fraction.
_MCD64 = 'MODIS/061/MCD64A1'
FIRE_RECORD = (2001, 2024)        # inclusive full-calendar-year window (record starts Nov 2000)


def _mcd64() -> ee.ImageCollection:
    """MCD64A1 monthly composites over ``FIRE_RECORD``.

    Note the product's masking convention (verified on Alaska): ``BurnDate`` is
    **masked** wherever a pixel did NOT burn that month — it is not a stored 0 —
    so ``BurnDate >= 1`` is unmasked *only* at burned pixels. The ``QA`` band, by
    contrast, is present over the whole product domain (land+water where the
    algorithm ran; ``QA & 1`` flags land), so coverage/never-burned must be
    established from ``QA``, not from ``BurnDate``."""
    y0, y1 = FIRE_RECORD
    return ee.ImageCollection(_MCD64).filter(ee.Filter.calendarRange(y0, y1, 'year'))


def _coverage(col: ee.ImageCollection) -> ee.Image:
    """1 where MCD64A1 reported in the domain (``QA`` present in >=1 month), else
    masked. Distinguishes never-burned ground (-> censored value) from truly
    off-product pixels (-> NaN)."""
    return col.select('QA').count().gt(0)


def burn_count() -> ee.Image:
    """Number of months in ``FIRE_RECORD`` with a detected burn — a right-censored
    fire count. Band ``'burn_count'``. ``0`` over covered-but-unburned ground
    (``BurnDate`` unmasked to 0 before summing); masked (-> NaN downstream) only
    off the product's coverage."""
    col = _mcd64()
    burned = col.map(lambda img: img.select('BurnDate').unmask(0).gte(1))
    return (burned.sum()
            .updateMask(_coverage(col))
            .rename('burn_count').toFloat())


def time_since_last_fire() -> ee.Image:
    """Years since the most recent detected burn, capped at the record length
    (right-censored). Band ``'tslf'``.

    Each monthly composite is stamped with its decimal year where the pixel
    burned (``BurnDate >= 1``, unmasked only there) and masked elsewhere; the
    temporal ``max`` is the most-recent burn year, and ``ref - that`` is the age
    in years. Pixels that never burned in the record are filled with the full
    record length (``ref - y0``) — the right-censored ceiling ("no fire since
    2001" != "never burned"). Pixels off the product's coverage (``QA`` absent)
    stay masked (-> NaN downstream)."""
    y0, y1 = FIRE_RECORD
    ref = float(y1 + 1)                       # decimal-year reference: start of the year after the record
    col = _mcd64()

    def year_where_burned(img):
        d = ee.Date(img.get('system:time_start'))
        dec_year = ee.Number(d.get('year')).add(d.getFraction('year'))
        burned = img.select('BurnDate').gte(1)                         # unmasked only at burned pixels
        return ee.Image.constant(dec_year).float().updateMask(burned).rename('yr')

    most_recent = ee.ImageCollection(col.map(year_where_burned)).max()  # masked where never burned
    age = ee.Image.constant(ref).subtract(most_recent)                 # years since last burn (masked where never burned)
    return (age.unmask(ref - y0)                                       # never-burned -> record-length ceiling
               .updateMask(_coverage(col))                            # off-coverage -> masked -> NaN
               .rename('tslf').toFloat())
