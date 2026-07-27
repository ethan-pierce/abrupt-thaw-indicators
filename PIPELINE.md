# PIPELINE

End-to-end provenance for the abrupt-thaw susceptibility pipeline: data sources,
the script→artifact DAG in execution order, and the final outputs. Describes the
current pipeline as built by the methods-cleanup rewrite (complete 2026-07-13).
The per-run **manifest** (git SHA, data hashes, seeds, hyperparameters) is stamped
at run time beside `model.json` and is the reproducibility key for a specific run.

## 1. Data sources

### Labels
- **Alaska Permafrost Thaw Database v2.0.0** — `data/Alaska_Permafrost_Thaw_Database_v2.0.0.csv`
  (`webb2026-thawdb`, `REFERENCES.md`; 19,540 rows, 93.21%/6.79% abrupt/non-abrupt).

### Features — public datasets (Google Earth Engine)
| Feature(s) | Source | GEE id | Reducer @ scale |
| --- | --- | --- | --- |
| Elevation, Slope, Aspect | USGS **3DEP 10 m** DEM | `USGS/3DEP/10m` | mean @ 10 m |
| 19 bioclimatic variables | **WorldClim v1** BIO | `WORLDCLIM/V1/BIO` | mean @ 1 km |
| SOC, Nitrogen, Clay, Sand, Silt, Bulk Density (6 depths each) | **SoilGrids** (ISRIC, 250 m) | `projects/soilgrids-isric/{soc,nitrogen,clay,sand,silt,bdod}_mean` | native @ 250 m (T35) |

> **WorldClim V1 is kept deliberately — do not "upgrade" to V2.1.** V1 supplies the
> climatological *level* (stable ~1960–90 normal); Daymet supplies the recent *trend*
> — different windows by design, not an inconsistency. V2.1 is not on the first-party
> GEE catalog (only a community `sat-io` asset or a 19-band local download), so a swap
> would either reintroduce third-party-asset fragility or add a large local file for a
> marginal gain. Decision 2026-07-14 (FABLE A3§6).

> **Soil texture — `Silt` is dropped in cleaning (T35).** Sand/Silt/Clay form a closed
> composition (sum ≈ 1000 g/kg), so one component is exactly redundant. `clean_feature_table.py`
> keeps Sand + Clay (best-conditioned pair on this ROI) and drops Silt; the datacube never
> builds it (soil layers are gated on the model's feature list). The raw silt bands are still
> sampled at the source above — the drop is a downstream cleaning step, not a sampling change.

### Features — formerly custom assets, now re-derived (no custom GEE assets)

> **Migration note (rebuild complete 2026-07-13):** access to the original
> `ee-abrupt-thaw` project was lost (2026-07-10). Compute has moved to the new project
> `abrupt-thaw-indicators` (`settings.EE_PROJECT`). All 13 old custom assets were
> confirmed **unreadable** (old project inaccessible) and no local copies existed, so
> they were **rebuilt from first-party sources — with zero custom uploaded assets** — to
> end the project-scoped-asset fragility for good. `ASSET_ROOT` has been **removed** from
> `settings.py` and from all code (TASKS T0). Two tracks:
> **(GEE)** `data/gee_features.py` — inline computation from account-independent public
> catalog datasets; and **(LOCAL)** `data/local_rasters.py` — nearest point/grid
> sampling of downloaded source rasters under `data/` (rasterio). Exact original
> derivation parameters are unrecoverable (build code was lost with the project), so
> re-derived layers use **documented, reconstructed** choices (fine for the v2.0.0
> rebuild — no byte-match to the lost assets is required). The GEE track's re-derived
> layers were pre-flight validated on Earth Engine before the first full build
> (T30, 2026-07-13): the **curvature port is verified** — its ZT kernels + mean-curvature
> expression reproduce a paraboloid's analytic `−2a` apex curvature exactly in NumPy
> (and are sign-robust to `convolve`'s flip convention), and on real 3DEP over Alaska
> it returns finite, physically plausible values (steep terrain ≫ flats; 2 km smoother
> than 500 m) via both the point and datacube (`sampleRectangle`) paths. SWE +
> SWE/precip/temp trends were migrated (2026-07-13) from live GEE point-sampling —
> which hangs on the deep temporal reduction — to a materialized local Daymet
> raster (`build_daymet_rasters.py`; `computePixels` tiled download, EPSG:3338 1 km),
> now sampled by both tracks via `local_rasters`. The fire representation was
> reworked (T36): the FIRMS `Maximum Fire Temperature` / `Fire Detected` pair
> (T4/T18) is **dropped** for a MODIS MCD64A1 fire *history* — `Time Since Last
> Fire` + `Burn Count` — likewise materialized to a local ~500 m raster
> (`build_modis_fire_rasters.py`), right-censored to the ~24-yr record.

| Feature | Track | Source (confirmed 2026-07-13) | Re-derivation / notes |
| --- | --- | --- | --- |
| Mean curvature (500 m, 2 km) | GEE | USGS **3DEP 10 m** (`USGS/3DEP/10m`) | reconstructed Zevenbergen–Thorne mean curvature (1/m) in EPSG:3338. The DEM is bilinearly resampled to `d = window / 2` (250 m or 1 km), so the outer cells of the 3×3 finite-difference neighborhood span the named 500 m or 2 km smoothing window; sampled at `d` in both paths |
| Height Above Nearest Drainage | GEE | **MERIT Hydro v1.0.1** (`MERIT/Hydro/v1_0_1`, band `hnd`) | raw height above nearest drainage (m), point-sampled at 90 m (rounded native ~92.77 m) in both paths; no 1 km averaging |
| Upstream Area | GEE | **MERIT Hydro v1.0.1** (`MERIT/Hydro/v1_0_1`, band `upa`) | raw upstream drainage area (km²), point-sampled at 90 m in both paths; no 1 km averaging and no canonical log transform (the linear baseline applies `log1p` within its own preprocessing) |
| Mean Annual SWE | LOCAL | **Daymet V4** (`NASA/ORNL/DAYMET_V4`, band `swe`), materialized to `data/daymet/daymet_v4_reductions_1km_3338.tif` by `build_daymet_rasters.py` — *resolved 2026-07-13* | inclusive 1991–2020 window: mean daily SWE within each calendar year, then mean across the 30 annual values; EPSG:3338 at native 1 km, nearest-sampled in both paths; materialized because live GEE point-sampling of the deep reduction hangs (T30) |
| Trend in SWE / precipitation / temperature | LOCAL | **Daymet V4** (`NASA/ORNL/DAYMET_V4`; `swe` / `prcp` / `tmax`), same materialized raster (`build_daymet_rasters.py`) — *resolved 2026-07-13* | inclusive 1991–2020 annual series: SWE = mean daily SWE, precipitation = annual sum, temperature = mean daily maximum temperature; per-pixel OLS slope against year via `ee.Reducer.linearFit()` (`scale` band); EPSG:3338 at native 1 km, nearest-sampled in both paths |
| Time Since Last Fire, Burn Count | LOCAL | NASA/USGS **MODIS MCD64A1** (`MODIS/061/MCD64A1`, monthly `BurnDate`), materialized to `data/modis_fire/mcd64a1_fire_history_500m_3338.tif` by `build_modis_fire_rasters.py` (T36) | inclusive 2001–2024 record: decimal years since the most recent detected burn and count of monthly composites with a detected burn; EPSG:3338 at ~500 m, nearest-sampled in both paths (the datacube samples at 1 km cell centers). **Right-censored:** no detected fire receives the 24-year ceiling and count 0, meaning "no fire since 2001," not never burned. Pixels outside the MCD64A1 QA domain are NaN (empirically ~11% of training points, concentrated above ~70°N) |
| Flammability Index | LOCAL | UAF **SNAP ALFRESCO** historical, CRU TS4.0 1900–1999 (`data/fetch_alfresco.py`) — *resolved 2026-07-13* | continuous 0–~0.02, EPSG:3338 1 km, nodata −9999; bilinear/nearest sample |
| Vegetation Mode | LOCAL | UAF **SNAP ALFRESCO** historical mode statistic, 1950–2008 (`data/fetch_alfresco.py`) — *resolved 2026-07-13* | **categorical** veg-type codes 0–8, EPSG:3338 1 km; **nearest** sample (never mean) |
| Land cover | LOCAL | **NLCD 2016 Alaska** ERDAS `.img`+`.ige` in `data/NLCD2016/` (user-provided) — *resolved 2026-07-13* | categorical NLCD codes, WGS84-Albers 30 m; windowed/nearest sample (8.4 B px — don't load whole array) |

> **All LOCAL-track sources resolved (2026-07-13).** ALFRESCO ×2 → historical CRU/observed runs
> (`data/fetch_alfresco.py`). NLCD → user-provided ERDAS `.img` in `data/NLCD2016/`.
> All are EPSG:3338 or WGS84-Albers, ready for rasterio point-sampling; raster
> binaries are git-ignored and regenerable from the fetch scripts. The reconstructed
> curvature, Daymet, MERIT Hydro, and MODIS choices are recorded in the methods table
> above; they are reproducible definitions for this rebuild, not byte-matches to the
> lost custom assets.
>
> **SNAP projected-climate features removed (2026-07-13).** The three
> `Projected summer/winter temp change` + `Projected precipitation change` layers
> (UAF SNAP AR5/CMIP5, 2090s − 2010s) were dropped: a future-scenario projection
> cannot causally drive a presently-observed thaw label (it acts only as a spatial
> proxy), and recasting it as a current-period trend would merely duplicate the
> observed Daymet trend already in the model. Temperature/precipitation are now
> sourced as WorldClim baseline level + Daymet observed trend only.
> `data/fetch_snap_projections.py` retired. `clean_feature_table.py` (and the diagnostics
> mirror) now drop these columns defensively, so a stale dirty table can't reintroduce
> phantom features the datacube never builds (which would crash `predict.py`).

### Masks (prediction domain / reliability)
- **Permafrost domain** — Obu et al. 2019 permafrost probability (PerProb 5.0),
  `data/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH/` (PANGAEA).
- **Categorical cross-check (optional)** — Brown et al. 1997 Circum-Arctic Permafrost
  and Ground-Ice map, `data/arctic-permafrost-map/` (NSIDC GGD318).

## 2. Execution order (script → artifact)

1. **`data/build_feature_table.py`** *(GEE; needs `ee.Authenticate()`, project `abrupt-thaw-indicators` via `settings.EE_PROJECT`)*
   reads the v2.0.0 database + all §1 feature sources → **`data/features_dirty.csv`**.
2. **`data/clean_feature_table.py`** reads `features_dirty.csv` → **`data/features_clean.csv`**.
   Fire is the MODIS MCD64A1 history pair (passes through untouched, T36 — no
   fill/derive); carries `Latitude`/`Longitude` as non-model columns; the dedup
   subset excludes coords.
3. **`models/train_xgboost.py`** reads `features_clean.csv` → **`models/model.json`** +
   **run manifest** + evaluation figures. Nested spatial CV over an equal-area
   (EPSG:3338) km-grid block-size sweep (operative 10 km blocks, buffer 0.0 km);
   selects on pooled-OOF AUC-PR (positive = Non-abrupt); `scale_pos_weight=1`;
   logistic + dummy baseline diagnostics through the same folds; persists CV config + seeds.
4. **`data/build_prediction_data.py`** *(GEE)* reads `roi.geojson` + §1 sources →
   **`data/prediction_data.nc`** (statewide **1 km** datacube, feature-name-driven;
   ~975 k cells over the ROI). **Terrain is served natively (T37):** a coarse `reproject`
   pyramid-aggregates the native derivative (the T37 probe measured slope
   collapsing to ~0.28× native at 4 km), so `sample_native` point-samples
   elevation/slope/aspect/curv-500 m at each 1 km cell centre via a chunked
   `reduceRegions` at the source's native scale — the identical construction the
   point path uses, so train and serve agree at native scale by construction.
   Curv-2 km and the 1 km bioclim layers are served by `reproject` (their native
   grid already ≈1 km — exact). **T35 generalized native serving to every source
   finer than the 1 km grid:** SoilGrids (250 m) and MERIT `upa`/`hnd` (~90 m) are
   now `sample_native` at their native scale (was `reproject(1 km)` for soil, and a
   `reduceResolution(mean)`-on-log for `upa`), so the datacube never reproject-averages
   a heavy-tailed feature — the canonical set stays raw/physical and train/serve parity
   is exact by construction. Aspect is served as northness/eastness (T32).
5. **`models/predict.py`** reads `prediction_data.nc`, `model.json`, Obu PerProb →
   **continuous log-evidence susceptibility surface** only. Emits the log-evidence index,
   Obu-masked; **no discrete classification**. It does **not** emit the
   AOA — reliability is a separate layer (T20/T21).
5b. **`models/aoa.py`** (the AOA reliability layer) runs *after* `predict.py` on the
   identical Obu-domain pixels. It scores each grid cell's importance-weighted
   dissimilarity index (DI) over a rank→training-CDF coordinate (mean|SHAP| weights) and
   emits `data/aoa.nc` (`DI` continuous — the headline reliability surface — plus a derived
   `inside_aoa` flag), `output/aoa_map.png`, `output/aoa_di_map.png`. Its threshold is
   anchored to CV performance by `diagnostics/aoa_calibration.py`
   (→ `models/aoa_threshold.json`): OOF AUC-PR holds ~15× the prevalence floor across the
   whole tested DI range, so the boundary is the edge of that measured-skill envelope
   (only the small fraction of cells more novel than anything CV tested is flagged).
6. **`models/shap_values.py`** reads `features_clean.csv` + CV config and **refits
   per-fold** → **pooled out-of-fold SHAP** outputs. The all-data `model.json`
   is deliberately **not** used here (OOF SHAP requires per-fold refits), so the
   *explained* model is **not the same fit** as the *mapped* model (`predict.py` maps
   all-data `model.json`) — both are correct, but Headline A (map) and Headline C (SHAP)
   do not come from one fit.

*Archived:* the training-lands path (`build_prediction_data_traininglands.py`,
`predict_traininglands.py`, `*_traininglands.*`) — see `archive/TASKS.md` T26.

## 3. Final outputs

- **`models/model.json`** + run manifest (git SHA, `features_clean.csv` hash, CV
  config + seeds, Obu/Brown product versions, selected hyperparameters).
- **Performance**: AUC-PR (positive = Non-abrupt) vs. block-size curve with across-fold
  spread + prevalence floor; AUC-ROC secondary; baseline diagnostics. *(no accuracy)*
- **Continuous log-evidence abrupt-thaw susceptibility surface** (statewide 1 km,
  masked by Obu permafrost domain) — the single headline map.
- **Area-of-Applicability reliability layer** (`data/aoa.nc`) — continuous rank→CDF
  dissimilarity index (headline) + derived extrapolation flag, threshold CV-calibrated
  (`diagnostics/aoa_calibration.py`). Emitted by `models/aoa.py`, **not** `predict.py`.
- **Pooled out-of-fold SHAP** — importance ranking, beeswarm, dependence plots.
- **`diagnostics/` re-run** — updated `/verify-ml` suite + regenerated `FINDINGS.md`
  verifying the new invariants (coord quarantine, buffer, OOF SHAP, baselines).
