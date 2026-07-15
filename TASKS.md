# TASKS — methods-cleanup + feature rebuild

Atomic, dependency-ordered task list, sorted relative to the **feature rebuild**
(the ~12 h `build_feature_table.py` point run + the `build_prediction_data.py`
datacube):

- **BLOCKING** — changes the *columns or values* the build emits, or is a
  correctness fix in a build script, or is a cheap pre-build probe whose result
  decides how the build is wired. Do these (or run the probe) **before** the
  overnight run; retrofitting after a multi-hour build is expensive.
- **DEFERRED** — operates on `features_clean.csv` and downstream (cleaning, CV,
  training, mapping, SHAP, docs). Runs **after** the rebuild.

New tasks **T31–T44** come from reconciling the FABLE.md agent recommendations
(grill-with-docs, 2026-07-14; decisions cross-referenced to FABLE agent/section).
Rationale for the original T1–T29 lives in `README.md` → "Methods cleanup"; target
pipeline in `PIPELINE.md`. An agent takes the next unchecked box whose *Depends* are
all met. Line numbers are current-code anchors — **verify before editing.**

**Class encoding unchanged** (`0 = Abrupt`, `1 = Non-abrupt`). The "Gradual" →
"Non-abrupt" change (2026-07-14, FABLE A2§2) is **glossary/label language only** —
the `0/1` encoding and `np.where(ThawType == 'Abrupt', 0, 1)` ground truth are
untouched. See CLAUDE.md glossary + SCOPE.md.

---

## BLOCKING — before / part of the feature rebuild

Ordered: pre-build probes first (their results shape the build), then the
column-changing wiring, then the dry-run gate before the overnight run.

- [x] **T37 — Terrain train/serve scale: probe first, don't coarsen blind.** [FABLE A1§1 / A2]
  ✓ 2026-07-14 **Probe (`diagnostics/probe_terrain_scale.py`): the mismatch is a
  *recompute*, and severe.** EE reproject to 4 km recomputes the derivative on a
  pyramid-aggregated DEM — slope collapsed to |·| median 0.31° vs. 1.08° native
  (ratio 0.28, corr 0.64); 500 m curvature decorrelated (corr ~0). A follow-up
  (`probe_native_serve.py`) showed `resample('bilinear'/'bicubic')` does **not**
  recover native for down-sampling — only literal native point-sampling does.
  **Decision (with Ethan, 2026-07-14): serve terrain natively AND upscale the
  prediction surface 4 km → 1 km** (see build_prediction_data `SCALE=1000` note).
  Point path keeps native sampling (unchanged — now correct). Datacube point-samples
  elevation/slope/aspect/curv-500 m at 1 km cell centres via
  `ee_sampling.sample_points_reduceregions_chunked` (chunked `reduceRegions` at
  native scale) — identical construction to the point path, so parity is exact by
  construction. Curv-2 km + bioclim served by `reproject(1 km)` (native grid ≈1 km,
  exact). SoilGrids (250 m) left on `reproject(1 km)` (mild; verified by T23).
  Validated: `diagnostics/probe_chunked_sampler.py` (chunked == single-call, max
  diff 0; off-grid → NaN). *(Per-feature parity confirmation is the DEFERRED T23.)*

- [x] **T35 — Systematic feature-transform audit.** [user-requested; generalizes A1§2]
  ✓ 2026-07-14 **Governing principle (with Ethan):** the canonical feature set —
  consumed by XGBoost *and* the datacube it scores — is **raw, physical, native-served,
  no logs**. A transform a specific model needs is *that model's* preprocessing (scoped
  to its own Pipeline), not a property of the shared data. Audit by bucket:
  - **(1) non-monotonic-for-correctness.** `Aspect`→Northness/Eastness (T32). Sand/Silt/Clay
    closure (sum ≈ 1000 g/kg) → **drop `Silt` canonically**, keep Sand+Clay: the
    best-conditioned pair on this ROI (corr(Sand,Clay)=+0.31; the discarded Sand-Silt
    axis is near-mirror at −0.945). No info lost (Silt recoverable), and it spares SHAP
    from splitting credit across three collinear columns. Re-swept the rest — signed
    quantities (curvature ×2, three trends, three projected changes), bounded ratios
    (Isothermality, Precip Seasonality), right-censored `Burn Count` — all monotonic-fine
    for a tree; nothing else qualifies.
  - **(2) must-precede-reprojection-averaging → EMPTY.** Resolution: generalize T37 — serve
    native wherever the native grid is finer than the 1 km serve grid, so the datacube
    never reproject-averages a heavy-tailed feature. This **reverts T34's `log(upa)`**:
    `upa` is now served raw/native like `hnd` (`gee_features.upstream_area()`, band `upa`,
    feature `Upstream Area`); the `reduceResolution(mean)`-on-log block + the `log(mean)`
    trap are deleted. SOC/Nitrogen/Bulk Density/Sand/Clay now `sample_native` at 250 m
    (was `reproject`). No pre-average transform survives anywhere.
  - **(3) linear-baseline (T13) / SHAP-readability → moved to T13's scope.** The live linear
    baseline logs the heavy-tailed positives (`Upstream Area`, SOC, Nitrogen, precip vars,
    `Mean Annual SWE`, `hnd`) + standardizes, inside its own Pipeline fit per CV fold.
    Nothing bucket-3 touches the canonical table or datacube; SHAP uses log *axes* at plot time.

  Baked into both paths: `clean_feature_table.py` (drop Silt), `gee_features.py`
  (`upstream_area`, raw), `build_feature_table.py` (rename → `Upstream Area`),
  `build_prediction_data.py` (`upa`+soil via `sample_native`), `smoke_feature_build.py`
  (probe renamed). Note: pure monotonic transforms are **no-ops for the XGBoost fit/ranking**.
  *Adjacent fix (done):* the three `Projected … change` features (SNAP, removed 2026-07-13)
  still lingered as columns in the stale `features_dirty.csv`→`features_clean.csv`, so a
  retrain would have embedded 3 phantom features the datacube never builds → `predict.py`
  would crash at `dataset_feature_names.index(name)`. Enforced the removal defensively at the
  cleaning layer (`clean_feature_table.py` + `diagnostics/_data.py`), regenerated the clean
  table (70→67 cols). *Still open:* per-feature parity confirmation is the deferred T23.

- [x] **T31 — Fix the datacube categorical double-flip.** [FABLE A3§1] *(datacube path only)*
  ✓ 2026-07-14 Land Cover and Vegetation Mode were `np.flipud`-ed **twice** while every
  other layer flips once — those two categoricals were vertically mirrored against the
  rest of the stack. Introduced by the T0 LOCAL migration (inner flip GEE-era, correct
  then; outer flip added for `sample_local`, inner left in). Removed the inner one-hot
  flips so both are `np.flipud(sample_local(...))` once, matching Flammability/Daymet.
  Added the `assert_local_orientation` guard (`build_prediction_data.py`): a
  vertical-mirror footprint test against the Flammability reference, run on each
  categorical's raw sample before one-hot. It witnesses Vegetation Mode directly (ALFRESCO
  nodata → informative footprint); Land Cover shares the identical sample+flip path so it's
  covered transitively (its own NLCD footprint is ~full → no nodata → auto-skipped).

- [x] **T32 — Aspect → northness/eastness.** [FABLE A1§2] ✓ 2026-07-14 Both paths now
  emit `Northness = cos(aspect)`, `Eastness = sin(aspect)` from natively-sampled aspect,
  with flats (**slope < 1°**) neutralized to `0` and NaN slope kept (mask → False); raw
  `Aspect` is dropped before save (point path) and never built (datacube). Point path:
  `build_feature_table.py` transform block after sampling. Datacube: computed in the T37
  terrain block from `sample_native(aspect)` + the shared slope array. Verified
  cos²+sin²≈1 off flats in `diagnostics/probe_chunked_sampler.py`.

- [x] **T33 — Add Yedoma as a feature.** [FABLE A1§7] ✓ 2026-07-14 The excess-ice/ground-ice
  control that mechanistically separates abrupt from non-abrupt thaw (replaces the rejected
  Obu-as-feature — Obu stays mask-only, T20). LOCAL track, source in-repo (IRYP v2, Strauss
  et al.; EPSG:3571). **Encoding decision (with Ethan): binary confirmed-presence** —
  `Yedoma = 1` inside a **confirmed** (tier-1, `conf_id // 10 == 1`) polygon, else `0`. Not
  ordinal: `conf_id`'s 2nd digit is the mapping *source*, not a geomorph subtype, and within
  the ROI "likely" is absent + "uncertain" is a 0.6% sliver, so confirmed-vs-everything ≈
  presence-vs-absence. **Prevalence check (decision gate):** confirmed yedoma = **15.6% of
  ROI area**; **25.1% of training points** (4907/19540), populated in both classes (Abrupt
  25.6%, Non-abrupt 18.4%), and the odds point the mechanistic way (non-abrupt rate 5.0%
  inside yedoma vs 7.4% outside, OR≈0.66) — clears the bar to keep. Shared helper
  `local_rasters.sample_yedoma(lons, lats)` (cached confirmed polygons + `sjoin` PIP;
  non-finite/off-ROI coords → NaN like `sample_points`); called identically by the point
  path (`build_feature_table.py`) and the datacube at its 1 km cell centres
  (`build_prediction_data.py`, gated on `'Yedoma' in feature_names`), so train/serve parity
  is exact by construction (T37 principle). Verified: helper reproduces the 4907/25.1% point
  prevalence; off-ROI −9999 fill → NaN; 975k-cell datacube call runs in ~0.5 s. *(The PIP
  machinery here also makes Brown ground-ice `CONTENT` a cheap future add.)*

- [x] **T34 — Add hydrological terrain features from MERIT Hydro.** [FABLE A1§7]
  ✓ 2026-07-14 GEE track, `MERIT/Hydro/v1_0_1` (official catalog; native ~92.77 m
  verified live). Two constructors in `gee_features.py`: `height_above_drainage()`
  (band `hnd`) and `upstream_area()` (band `upa`). Both paths emit
  `Height Above Nearest Drainage` + `Upstream Area`, both **raw and served natively**
  (point-sample at `MERIT_SCALE` like the 3DEP terrain, T37; datacube `sample_native`
  == point `reduceRegion`, parity exact — confirmed). **SUPERSEDED BY T35:** the original
  design served `log(upa)` reproject-averaged to 1 km (with an explicit
  `reduceResolution(mean)` on the native-pinned log, to avoid the verified `log(mean(upa))`
  trap — probe −2.59 vs correct −4.44). T35 removed that entirely: `upa` is now raw/native
  like `hnd` (no averaging → no log-order trap), and the log moved to the T13 linear
  baseline's own scope. Smoke gate (`smoke_feature_build.py`) PASSES: both features finite
  at all sample points.

- [x] **T36 — Fix fire representation (Package B).** [FABLE A1§6 / A3§4] ✓ 2026-07-14
  **Dropped** `Maximum Fire Temperature` (FIRMS `T21`, peak brightness of one detection)
  and `Fire Detected` (near-constant at 1 km) — reverting T4/T18. **Kept** Flammability
  Index. **Added** MODIS `MCD64A1` **Time Since Last Fire** + **Burn Count** via new
  `gee_features` constructors, materialized to a local ~500 m raster
  (`build_modis_fire_rasters.py`, the `build_daymet_rasters.py` pattern; datacube
  resamples to 1 km) and read by both tracks through `local_rasters.MODIS_FIRE_BANDS`.
  `clean_feature_table.py`/`diagnostics/_data.py` no longer derive `Fire Detected` or
  fill the old temp column; `smoke_feature_build.py` probes the new raster (conditional,
  like Daymet). Both features **right-censored** to the ~24-yr record (`FIRE_RECORD`
  2001–2024; "no fire since 2001" ≠ never-burned) and an **>70°N QA coverage gap**
  (fire features NaN for ~11% of points, all Arctic-coast) documented in
  `gee_features.py` / PIPELINE / SCOPE / README. Live-GEE verified: burned→age+count,
  unburned land→censored 24/0, off-coverage→NaN. Raster materialization + full build
  run overnight (T39).

- [x] **T39 — Build robustness + pre-build GEE dry-run.** [FABLE A2§7] ✓ 2026-07-14
  **(1)** `build_feature_table.py` reworked for full crash-safety: every feature (BOTH
  tracks — the LOCAL track previously had *no* guards) is added via `try_add`, so a
  per-feature failure is recorded in `failed_features` and printed, never aborting the run;
  the report + `features_dirty.csv` write run in a `finally` (`finalize()`), so hours of
  work are never lost to a late failure; init is non-interactive-safe (cached token first,
  `ee.Authenticate()` fallback only — no browser prompt hangs the overnight run). Verified:
  end-to-end 8-point self-test (via new `FEATURE_BUILD_LIMIT`/`FEATURE_BUILD_OUT` hooks)
  builds all 86 columns incl. T32 Northness/Eastness (raw Aspect dropped), and a negative
  test (broken MODIS path) confirms the two fire features are named in the report while the
  build continues and STILL writes the CSV.
  **(2)** New `dry_run_gee.py` validates auth/bands/schema for the GEE compute (3DEP T37
  terrain probe, MERIT hnd/upa, bioclim, one SoilGrids band per property) over 400 ROI-
  spread points AND reports statewide NaN fractions — **terrain 0.5%, soil 11.6%** (the
  empirical 3DEP/SoilGrids coverage caveat, SCOPE); it also existence-checks every LOCAL
  source. Gate PASSES. `smoke_feature_build.py` gained a Yedoma probe.
  **Prerequisite resolved:** the MODIS MCD64A1 fire raster (deferred from T36) is now
  materialized (`build_modis_fire_rasters.py`, 3232×3140 @ 500 m, tslf 0.33–24 yr, burn 0–8).
  **Heads-up (not fixed):** `USGS/3DEP/10m` is now a *deprecated* GEE asset (superseded by
  `USGS/3DEP/10m_collection`); it still serves data, and both the build and the datacube use
  `gee_features._DEM_ID`, so leaving it consistent is safe for this rebuild — a swap is a
  separate task. No `clean_feature_table.py` hardening — cheap to re-run against
  `features_dirty.csv`.

- [x] **T46 — Statewide extraction footprint (replaces `roi.geojson` contents).** [T20 grill 2026-07-15]
  ✓ 2026-07-15 The datacube rebuild must target **statewide Alaska**, not the stale
  `data/roi.geojson` (single North-Slope polygon, lat 68–71.4°N, unchanged since the datacube-
  prototype commit; didn't even match the old cube's extent, and contradicted the statewide
  north-star + the training span 56.9–71.4°N). **`roi.geojson` was NOT deleted** — it has **four**
  live consumers (`build_prediction_data.py` hard-clips to it; `build_daymet_rasters.py` /
  `build_modis_fire_rasters.py` union its bbox with the ThawDB point bbox; `dry_run_gee.py` scatters
  QA points in it), so it defines the datacube/raster domain. The stale North-Slope contents were a
  **landmine that hadn't gone off**: the raster builders' point-bbox union subsumed the tiny ROI
  (→ statewide rasters already), `dry_run` makes no product, and the old cube predated the stale
  file — only the *next* datacube rebuild would have been corrupted. **Fix:** rewrote `roi.geojson`'s
  *contents* to the Alaska land boundary — `TIGER/2018/States` NAME='Alaska' geometry `.intersection(
  Rectangle([-170,51,-141,72], geodesic=False), maxError=1000)`. Verified live: the raw AK geometry
  wraps the antimeridian (−179.2°…+179.9° via the Aleutians) but the intersection collapses cleanly
  to [-170,-141]×[52.7,71.4] (25 polygons: mainland + in-bbox islands, no wrap), contains Utqiagvik /
  Seward Pen / Fairbanks / Anchorage, and the Obu permafrost domain sits well inside it (visual check,
  throwaway figure). `build_prediction_data.py` reads the file and coerces to `ee.Geometry` (the
  strip-tiling `extract_data_array` needs a Geometry, not a FeatureCollection); `predict.py`'s map
  extent moved to the persisted per-cell lon/lat (T20), so it no longer reads the file. **One
  statewide source of truth** for all four consumers. Old North-Slope polygon backed up in the
  session scratchpad. *Depends:* — *Done:* build reads the statewide boundary; `predict.py` decoupled.

---

## DEFERRED — after the rebuild (operate on `features_clean.csv` / downstream)

- [x] **T43 — CV buffer from the empirical autocorrelation range.** [FABLE A2§3]
  ✓ 2026-07-15 `train_xgboost.py` (**not** `spatial_cv.py`) fixed `BUFFER_KM = 1.0` [B5c],
  applied across all `SWEEP_CELL_KM` sizes. Two probes, both with a matched-count random-
  removal CONTROL to separate leakage from data loss, positive = Non-abrupt, no nominal-
  scale floor:
  - **Random-split geometry (`diagnostics/leakage_decay.py`, figure `leakage_decay.png`):**
    r=0 AUC-PR 0.904 (leaky ref); the leakage-specific gap is contiguous only through
    **~2 km** (+~0.08), then dissolves into data-depletion noise — a 1 km buffer already
    strips 63% of train points, 2 km strips 86%, so this dispersed-test geometry can't
    yield a plateau (the pool depletes first; the naive plateau detector was fixed to
    reject the collapsed chance floor). Diagnostic-only; wrong geometry to *size* the
    operative buffer (which feeds block CV, not a random split).
  - **Block-CV geometry (`diagnostics/block_cv.py`, figure `block_buffer_decay.png`):**
    the operative geometry — `albers_grid`/`cell_km=10`/5-fold/seed-42, fixed
    `leakage_decay` estimator (operative hparams stale pre-retrain + circular). Buffer
    sweep 0–15 km × 1 km: block-holdout AUC-PR **0.789** at buffer 0 (vs 0.904 leaky — the
    **block structure**, not the buffer, is what removes the ~0.115 of leaked AUC-PR), and
    the targeted curve stays **flat** with the **targeted-vs-control gap never exceeding
    0.018** (< 0.02) out to 15 km / 78% removed. **⇒ no near-seam leakage survives block
    holdout; `BUFFER_KM = 0` is defensible (data-driven, no floor).**
  **Set `BUFFER_KM = 0.0`** in `train_xgboost.py`. *(Not the explanation for the old
  random-split near-perfect discrimination — that predates the buffer; separate leakage
  question → SCOPE.)* Buffer-sensitivity sweep reported in both figures; the requested
  1/2/5/10 km readout is in the block-CV table.

- [ ] **T23 — Train/serve parity gate (broadened).** [FABLE A2 / A1§5 / A3§5; absorbs T42 + the T37 tail]
  ⚠ **BLOCKED (2026-07-15): the serve side is still pre-rebuild.** Only the *train* side of
  the rebuild is done — `features_clean.csv` is fresh (71 model features: Northness/Eastness,
  Yedoma, MERIT `hnd`/`upa`, MODIS `Time Since Last Fire`/`Burn Count`, no `Silt`). The
  datacube `data/prediction_data.nc` (and `models/model.json`) are still the **old 49-feature,
  4 km schema** (`scale=4000`, 294×862): they carry `Aspect`, `Silt`, `Maximum Fire Temperature`,
  `Projected precipitation change` and lack every new feature. So train and serve are on
  **disjoint feature sets** and a per-feature parity comparison is not yet meaningful.
  **Unblock:** re-run `data/build_prediction_data.py` (the GEE overnight run — emits the new
  1 km datacube from the completed feature-table rebuild) and retrain (`models/train_xgboost.py`);
  T23 runs once train and serve share the new schema.
  *Depends:* rebuild (**datacube half still outstanding** — feature-table half done). *Done when:* a per-feature **training-column-vs-datacube-pixel
  distribution-parity** check is documented for **every** feature (not soil-NaN only),
  including: soil-NaN reproduction; **the soil 250 m→1 km `reproject`-averaging** left
  unfixed under T37 (the one remaining terrain/soil scale gap — confirm it is as mild as
  assumed); terrain (now served natively both sides → expect near-exact parity, the T37
  construction check); and the **land-cover/veg category-set subset check** (report any
  class present statewide but absent from training points, and the area affected — silent
  reference-bucket absorption). No change to one-hot construction.

- [ ] **T20 — Obu domain mask.** [G17; design resolved via grill 2026-07-15]
  Replace the arbitrary "≥50% features non-NaN" keep (`predict.py:94-97`) with a permafrost-
  domain mask from Obu PerProb (`local_rasters.OBU_TIF` / `sample_points`). **Design (with Ethan):**
  - *Concept-validity mask, not reliability* — off-permafrost, abrupt-vs-non-abrupt thaw is
    **undefined** (not merely uncertain); the model saw ~0 non-permafrost training points
    (0.4% below PerProb 0.01, median 0.924) so it **cannot self-mask** — scored off-domain it
    emits a meaningless likelihood ratio between two thaw modes that don't exist there.
    Reliability stays a **separate** layer (T21/AOA).
  - *Threshold `PerProb > 0`* — keep the whole permafrost domain incl. isolated. Obu assigns
    **exactly 0** to modeled non-permafrost (9.7% of finite pixels statewide) and small
    positives to isolated permafrost, so no epsilon is needed. Higher thresholds **rejected**:
    PerProb is label-entangled (non-abrupt median 0.366 vs abrupt 0.935), so cutting high
    amputates the minority class's home range and biases the map toward "all abrupt."
  - *Hard binary mask, NO soft-weighting* — down-weighting by PerProb would systematically
    suppress the minority (non-abrupt) class, reintroducing the bias the low threshold avoids.
    PerProb never multiplies the surface (may be persisted as passive metadata only).
  - *Keep rule* `keep = (PerProb > 0) AND (≥1 feature non-NaN)` — deletes the arbitrary 50%
    gate; the minimal all-NaN guard refuses to paint a base-rate pixel from zero evidence
    (XGBoost returns a finite base score on all-NaN input, not NaN).
  - *Where* — the build persists per-cell **lon/lat** as datacube coords (fixes the real
    defect: the cube is ungeoreferenced, x/y are bare indices); `predict.py` samples Obu at
    those coords via `sample_points`. Obu stays **mask-only** (not a feature — 2026-07-14),
    never entering the feature machinery.
  - *Applies to the saved products* (NaN outside domain), not just the display figures — the
    current masking is figure-only, so `susceptibility.nc` presently carries values over ocean.
  *Depends:* T19 (done), T46 (done — statewide footprint), **rebuild** (1 km statewide datacube —
  outstanding, per T23). **Code implemented + smoke-tested 2026-07-15** (build persists lon/lat;
  `predict.py` samples Obu + masks saved products; verified end-to-end on a synthetic cube:
  keep = PerProb>0 AND ≥1 feature, off-domain → NaN, zero finite values off-domain). *Done when:*
  off-permafrost pixels are NaN in `susceptibility.nc` and the other saved products (not
  figure-only) **on the rebuilt statewide 1 km datacube**.

- [ ] **T21 — AOA mask.** [G18] Importance-weighted dissimilarity-to-training mask with a
  CV-derived threshold, output as a reliability layer. *Depends:* T14 (done), T19 (done).
  *Done when:* an extrapolation-flag raster is produced.

- [x] **T22 — Remove discrete classification.** [G19] ✓ 2026-07-15 Stripped the discrete
  class path from `predict.py`: removed `DECISION_THRESHOLD` (config + print + attr), the
  `predictions = (probabilities < threshold)` binarization, `predictions_2d` + its mask, the
  two abrupt/non-abrupt count prints, the `'prediction'` variable + `prediction_description`
  attr from `output_ds`, the `prediction_classes.nc` write, and the classification-map figure
  (`prediction_classification_map.png`). Deliverable is now the continuous log-evidence
  surface (`susceptibility.nc` + map) with probability kept as an explicit *diagnostic*
  (`predictions.nc`, `prediction_probabilities.nc`, probability map) — the discrete class is
  the only thing gone, per G19. Also `git rm`'d the stale tracked `data/prediction_classes.nc`
  (old-cube orphan) and updated the README's three stale "classification map" deliverable
  mentions (line 12, tree, run note, To-Do #8). Archived training-lands path left untouched
  (T26). Verified: `py_compile` clean, no dangling refs (`predictions_2d`/`pred_ds`/
  `map_output_path2`/`DECISION` all gone). *Depends:* T19 (done). *Done when:* no discrete
  class artifact is written. *(End-to-end run deferred to the rebuilt cube — predict.py's next
  natural run under T20/T23; running now would clobber real `data/` mid-rebuild.)*

- [x] **T40 — Class-label sweep "Gradual" → "Non-abrupt".** [FABLE A2§2 / glossary 2026-07-14]
  ✓ 2026-07-15 Swept every "Gradual" class-1 label across **our** code + methods docs to
  "Non-abrupt": `train_xgboost.py` (headline plot ylabel, prevalence print, per-fold metric
  strings, docstring — plus the internal `test_gradual`/`train_gradual` dict keys → underscore
  form `test_non_abrupt`/`train_non_abrupt`, which also flow into `cv_sweep_results.json`),
  `predict.py` (class-map colorbar ticklabels, prediction-count print, netCDF
  `prediction_description`, log-evidence/cmap comments), `shap_values.py`, `spatial_cv.py`,
  four `diagnostics/*.py`, `clean_feature_table.py` (encoding comment), and `README`/`PIPELINE`/
  `FINDINGS`. **Left `REFERENCES.md` alone** — its "abrupt-vs-gradual thaw *mode*" usages
  describe the external-literature concept (the glossary keeps "gradual thaw" valid as a
  concept), not the class label. Encoding untouched (`np.where(ThawType=='Abrupt',0,1)`).
  Verified: all files compile, no residual "gradual" in code, and a redirected-output smoke
  (real artifacts untouched) prints "test Non-abrupt …" with the renamed keys serialized.

- [ ] **T41 — Grouped SHAP over emergent groups.** [FABLE A1§4 / A2§6 / A3§2]
  ⏳ **Machinery built + design settled (grill 2026-07-15); authoritative run deferred.**
  New `models/shap_groups.py` reuses the canonical `shap_values.pooled_oof_shap` (per-fold
  refit + held-out TreeSHAP) and adds the grouping layer. **Purpose fixed as (a):** a
  de-cluttered, geoscientist-legible indicator-**family** importance ranking so credit isn't
  split across the ~70 partly-redundant columns — NOT an a-priori "landscape regime" claim
  (families interpreted post-hoc), and NOT the mechanism-vs-lake-proxy question (SCOPE defers
  that). **Settled design:** grouping basis = **feature-space** Spearman (a tree scatters
  credit erratically across near-duplicate cols, so SHAP-space could fail to group them;
  25/44 continuous cols have a |ρ|>0.8 partner); distance = **1−|Spearman|** (anti-correlated
  cols are still redundant — fire pair ρ=−1.00, thermal continentality spans ±0.94); linkage
  = **complete** (cut at t ⇒ within-family |ρ|≥1−t); cut = the **auto-detected natural gap**
  in merge heights (emergent, ~|ρ|≈0.55 → 19 continuous families on the prototype); one-hots
  **collapsed to source** (Land Cover, Vegetation Mode) with **Yedoma standalone** (one-hot =
  one variable, definitional redundancy). Grouped contribution/point = **Σ signed member
  SHAP** (exact additivity); importance = mean|Σ|, Abrupt-oriented. Outputs: dendrogram +
  grouped-importance bar + per-family contribution box + `shap_families.json`. **Smoke-tested
  end-to-end** (`SHAP_GROUPS_SMOKE=1` → `output/_smoke/`, gitignored, non-authoritative).
  *Deferred to the operational run (feature set must be final — i.e. **post-T23**, since a
  parity-driven feature drop would force a retrain and move the SHAP):* the authoritative
  numbers + two case-by-case curation calls judged against real importances — (1) keep the
  4-member "alpine relief" [Elevation|Slope|HND|SWE] fused or split terrain from elev/snow;
  (2) keep `Trend in SWE` in thermal continentality or move it with the other trends. Then
  rename the auto-tagged provisional family labels for the manuscript. Full design recorded in
  memory `t41-grouped-shap-design`.
  *Depends:* T25 (done), retrain (done 2026-07-15 15:05 — operative model already current),
  **feature-set lock (T23)** for the authoritative numbers. *Done when:* the authoritative
  grouped SHAP story is reported over the emergent families (§ above), curation resolved.

- [x] **T44 — Contradictory-label ceiling diagnostic.** [FABLE A3§3] *Depends:* rebuild
  (train half — `features_clean.csv` — done). ✓ 2026-07-15 `diagnostics/contradictory_labels.py`
  (consumes the `_data.load` deduped model matrix): groups rows by exact feature-identity
  (NaN==NaN, matching the pipeline dedup), flags feature-identical / label-disagreeing
  groups. Because `clean_feature_table.py` dedups on (features, Class) jointly, each such
  group survives as a clean 1:1 pair (one Abrupt, one Non-abrupt) — asserted. **Result: only
  4 contradictions** (8 rows, 0.04%; **0.36% of the 1107 minority**), pair members 2–6 m
  apart (confirms the shared-source-pixel mechanism). Ceilings: **accuracy 0.99979**,
  **oracle AUC-PR 0.99999** (chance 0.0574). **Finding (honest negative):** exact
  feature-identity contradictions are *negligible* — they do **not** bound separation and do
  **not** explain the GBM's gap-to-1.0. The separation limit is **soft feature-space overlap**
  (near-but-not-identical opposing labels), which exact-match can't see. NOTE: dedup collapses
  same-(feature,label) multiplicity, so the deduped 1:1 view is the operative ceiling for the
  fitted model (it trains/scores on the same table).
  **UPDATE 2026-07-15 (repeated-CV, `diagnostics/repeated_cv.py`):** the "narrow GBM-vs-logistic
  margin" this note referenced was a **single-partition + hyperparameter-selection-variance
  artifact**, not real. The headline sweep's +0.003 @10 km used per-fold *selection* for both
  models on one CV partition; selection there helped the linear model (best-fixed 0.785 → selected
  ~0.81) and did **not** help the tree (fixed operative 0.852, min 0.819 across 20 reshuffles >
  the single selected 0.815). Under fair fixed-config CV repeated over 20 block→fold reshuffles,
  the operative-scale margin is **+0.076 ± 0.019** (≈4σ outside partition noise), holding +0.07–0.09
  across 5–100 km and shrinking to +0.034 only at 200 km. **So separation is mostly linear BUT with
  a real, stable ~0.07-AUC-PR non-linear/interaction component the tree captures** — refines, not
  contradicts, the soft-overlap ceiling above.
  *Follow-up (offered, not built):* a near-neighbour / feature-overlap ceiling to quantify the
  soft limit — needs a feature-distance metric decision.

- [x] **T45 — Fix logistic baseline numerical blow-up on the new feature set.** [observed 2026-07-15]
  ✓ 2026-07-15 **Not a bug — the deferred T35 bucket-3 decision (the linear baseline owns
  its own preprocessing) had never been wired in.** The baseline was `median-impute →
  standardize → lbfgs-logistic`, missing the log-compression. Root-caused in two layers:
  - **The flood = the lbfgs *fit*.** On this near-separable data lbfgs spends ~100
    line-search iterations, each emitting a transient overflow/invalid-`matmul` warning
    (the *final* coef is small & finite — |coef|≤3.45 — so the warnings are benign
    optimizer noise, not divergence). Switched the baseline solver to **`liblinear`**
    (coordinate descent): identical fit (AUC-PR 0.889, same coef norm) in <10 iterations,
    fit is warning-free.
  - **Preprocessing wired per T35, split by column type** (`logistic_builder`
    `ColumnTransformer`): heavy-tailed non-negative continuous (`LOG_BASELINE_COLS` — MERIT
    `hnd`/`upa`, precip *amounts*, SWE, SOC, Nitrogen) get `log1p → median-impute →
    standardize`; other continuous get `median-impute → standardize`; **binary one-hots
    (Land Cover / Veg Mode / Yedoma, detected by value) are NOT standardized** (dividing a
    rare 0/1 col by its tiny σ inflated its lone `1` to ~139σ). NaN handled cleanly (median
    for continuous, constant-0 for one-hots).
  - **Residual 3 warns/predict = a known benign numpy SIMD `matmul` false positive** (the
    decision fn is finite, range −12…+16). Scoped an `np.errstate(divide/over/invalid=
    'ignore')` around the baseline only via a thin `_QuietLinearBaseline` fit/predict
    wrapper — XGBoost untouched.
  **Verified:** end-to-end `TRAIN_SMOKE=1` run exits 0 with **zero** `RuntimeWarning`s and a
  sensible baseline (pooled-OOF AUC-PR ~0.72 vs dummies ~0.05). `model.json` (XGBoost,
  NaN-native) is unchanged by this — only the baseline comparison is now trustworthy.
  *Follow-up:* a full (non-smoke) retrain will refresh the **persisted** baseline AUC-PR in
  `run_manifest.json` / the sweep log (the last full run predates this fix).

- [ ] **T28 — Update `/verify-ml` + regenerate FINDINGS.** [H20.2] *Depends:* T14 (done),
  T19 (done), T25 (done), + rebuild. *Done when:* the `diagnostics/` suite is re-pointed at
  the new invariants (coord quarantine, empirical buffer, OOF SHAP on held-out, baselines,
  parity gate, contradictory-label ceiling) and `FINDINGS.md` is stamped to the new pipeline.

- [ ] **T29 (remainder) — Document reconstructed GEE-track params.** Record the curvature
  smoothing window, Daymet year range/aggregations, and the new MERIT/MODIS choices in the
  methods table. *Depends:* T34, T36. *Done when:* the methods table reflects the rebuilt
  feature set. *(Acquisition half of T29 done 2026-07-13.)*

---

## RESOLVED

- **T30 — Heavy inline-GEE features hang at full-N point sampling. → MOOT (2026-07-14).**
  The two remaining live heavy point-samples are gone: SWE + the 3 trends were migrated to
  the materialized local Daymet raster (`build_daymet_rasters.py`, commit `5dffcbf`), and
  `Maximum Fire Temperature` is **dropped** by T36. No live deep-temporal reduction remains
  at points (MODIS burn-history, T36, is materialized like Daymet, not point-sampled). The
  T39 dry-run confirms no heavy point-sample survives. Priority override lifted.

---

## COMPLETED (T0–T29, compact — full historical done-notes in git @ `ecb7a94`)

- [x] **T0** — Re-establish GEE project & rebuild feature sourcing with **zero custom
  assets** (two tracks: `gee_features.py`, `local_rasters.py`; `ASSET_ROOT` removed). ✓ 2026-07-13
- [x] **T1** — Equal-area (EPSG:3338) km-grid block method in `assign_blocks`. ✓
- [x] **T2** — Nested-fold helper (inner folds never touch outer test blocks). ✓
- [x] **T3** — Per-fold Non-abrupt (class-1) count reporting. ✓
- [x] **T4** — Fire encoding (`Fire Detected` binary + real-or-NaN temp). ✓ **← reverted by T36**
- [x] **T5** — Dedup excludes coordinates. ✓
- [x] **T6** — Carry `Latitude`/`Longitude` through as non-model columns. ✓
- [x] **T7** — Load + quarantine coords out of `X`. ✓
- [x] **T8** — Nested spatial CV over a block-size sweep (buffer 1 km). ✓
- [x] **T9** — Selection on pooled-OOF AUC-PR (positive = Non-abrupt). ✓
- [x] **T10** — `scale_pos_weight = 1` (no imbalance reweighting). ✓
- [x] **T11** — Grid breadth + named seeds; CV config persisted. ✓
- [x] **T12** — Headline AUC-PR-vs-block-size curve; accuracy removed. ✓
- [x] **T13** — Dummy + penalized-logistic baselines through the same folds. ✓
- [x] **T14** — Operative all-data refit → `models/model.json`. ✓
- [x] **T15** — Calibration demoted (block removed in the T8–T12 rewrite). ✓
- [x] **T16** — Run manifest (git SHA, data hash, seeds, hparams, product versions). ✓
- [x] **T17** — Removed mislabeled "CV F1" block. ✓
- [x] **T18** — `Fire Detected` datacube layer. ✓ **← reverted by T36**
- [x] **T19** — Log-evidence output (`logit(P) − logit(π_sample)`, 0 = neutral). ✓
- [x] **T24** — SHAP canonical plumbing (no independent re-split; loads CV config). ✓ 2026-07-13
- [x] **T25** — Pooled out-of-fold SHAP (per-fold refits; `model.json` intentionally unused). ✓ 2026-07-13
- [x] **T26** — Archived training-lands path. ✓ 2026-07-13
- [x] **T27** — Retired calibrated artifacts. ✓ 2026-07-13
- [x] **T29** — Acquired LOCAL-track source rasters (SNAP/ALFRESCO/NLCD; SNAP projections retired). ✓ 2026-07-13
