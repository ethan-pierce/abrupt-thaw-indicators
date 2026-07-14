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

- [ ] **T35 — Systematic feature-transform audit.** [user-requested; generalizes A1§2]
  For every feature decide whether it needs a transform, in three buckets: **(1)
  non-monotonic-for-correctness** (circular → cos/sin; signed quantities); **(2)
  must-precede-reprojection-averaging** (heavy-tailed → `log` *before* the mean — the
  datacube `reproject`-averages any feature coarser than its 1 km grid, a non-tree op;
  post-T37 this is soil + `log(upa)`, since terrain is now served natively); **(3)**
  linear-baseline (T13) / SHAP-readability
  only. *Depends:* — *Done when:* the audit is recorded and every bucket-(1)/(2) transform
  is baked into **both** build paths. **Owns the sand+silt+clay closure** (exact
  compositional dependence, sum ≈ 100%): drop one component or use isometric log-ratios.
  T32 (aspect) and T34 (`log upa`) are already-decided instances. Note: pure monotonic
  transforms are **no-ops for the XGBoost fit/ranking** — payoff is buckets (1) and (2).

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
  verified live). Both constructors in `gee_features.py`: `height_above_drainage()`
  (band `hnd`) and `log_upstream_area()` (`upa.log()`, band `log_upa`; `upa` strictly
  positive, min = one native cell, so `log` is safe). Both paths emit
  `Height Above Nearest Drainage` + `Log Upstream Area`. **`hnd` served natively** in
  both paths (raw stored height → point-sample at `MERIT_SCALE` like the 3DEP terrain,
  T37; datacube `sample_native` == point `reduceRegion`, parity exact — confirmed).
  **`log(upa)` reproject-averaged to 1 km** (heavy-tailed, finer than the grid; T35
  bucket 2): the log is baked into the shared image so the datacube averages on the log
  scale. Caveat + a verified trap documented in the `log_upstream_area` docstring — a
  plain `reproject` of the `.log()` image silently computes `log(mean(upa))` (probe:
  −2.59 vs. correct mean-of-log −4.44), so the datacube aggregates with an explicit
  `reduceResolution(mean)` on the native-pinned log. Smoke gate (`smoke_feature_build.py`,
  probes added) PASSES: both features finite at all sample points.

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

- [ ] **T39 — Build robustness + pre-build GEE dry-run.** [FABLE A2§7]
  *Depends:* T32, T33, T34, T36 (needs the final column set). *Done when:* **(1)** the
  build keeps **continue-on-failure** (no hard abort — protect the overnight run) and its
  end-of-run report loudly names every feature that raised or came back all-NaN, verified
  to cover the new columns; **(2)** a **pre-build GEE dry-run** over a few hundred points
  validates auth/bands/schema for the new GEE compute (MERIT + T37 probe) and reports
  statewide **NaN fractions** for terrain/soil (the empirical half of the 3DEP/SoilGrids
  coverage caveat, SCOPE). No `clean_feature_table.py` hardening — it is cheap to re-run
  against the preserved `features_dirty.csv`.

---

## DEFERRED — after the rebuild (operate on `features_clean.csv` / downstream)

- [ ] **T43 — CV buffer from the empirical autocorrelation range.** [FABLE A2§3]
  `spatial_cv.py` fixes `BUFFER_KM = 1.0`; with 97% of points within 5 km of a neighbor
  and features with multi-km spatial support (e.g. the 2 km curvature window), near-seam
  leakage may inflate the headline AUC-PR. *Depends:*
  rebuild. *Done when:* `diagnostics/leakage_decay.py` measures the leakage-decay range
  (where OOF performance stops dropping as the buffer grows), the operative buffer is set
  to that range with **no nominal-scale floor** (let the data decide — the 1 km serve grid
  is a *resampling* grid, not a native floor; terrain is served at native scale), and a buffer-sensitivity sweep
  (1/2/5/10 km) is reported. *(Not the explanation for the old random-split near-perfect
  discrimination — that predates the buffer; separate leakage question → SCOPE.)*

- [ ] **T23 — Train/serve parity gate (broadened).** [FABLE A2 / A1§5 / A3§5; absorbs T42 + the T37 tail]
  *Depends:* rebuild. *Done when:* a per-feature **training-column-vs-datacube-pixel
  distribution-parity** check is documented for **every** feature (not soil-NaN only),
  including: soil-NaN reproduction; **the soil 250 m→1 km `reproject`-averaging** left
  unfixed under T37 (the one remaining terrain/soil scale gap — confirm it is as mild as
  assumed); terrain (now served natively both sides → expect near-exact parity, the T37
  construction check); and the **land-cover/veg category-set subset check** (report any
  class present statewide but absent from training points, and the area affected — silent
  reference-bucket absorption). No change to one-hot construction.

- [ ] **T20 — Obu domain mask.** [G17] Soft-mask/weight by Obu PerProb
  (`data/Obu2019/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif`; `local_rasters.OBU_TIF` /
  `sample_points`), resampled to the 1 km Albers grid; replaces the feature-validity keep
  (`predict.py:94-96`). Obu is **mask-only** (not a feature — decision 2026-07-14).
  *Depends:* T19 (done). *Done when:* off-permafrost pixels are masked/down-weighted.

- [ ] **T21 — AOA mask.** [G18] Importance-weighted dissimilarity-to-training mask with a
  CV-derived threshold, output as a reliability layer. *Depends:* T14 (done), T19 (done).
  *Done when:* an extrapolation-flag raster is produced.

- [ ] **T22 — Remove discrete classification.** [G19] Delete the discrete class output
  (`prediction_classes.nc`, classification map, `DECISION_THRESHOLD`) so the deliverable is
  the continuous log-evidence surface only. *Depends:* T19 (done). *Done when:* no discrete
  class artifact is written.

- [ ] **T40 — Class-label sweep "Gradual" → "Non-abrupt".** [FABLE A2§2 / glossary 2026-07-14]
  *Depends:* — *Done when:* confusion-matrix labels, SHAP plot titles, and metric-report
  strings say "Non-abrupt" (matching the corrected glossary). Cosmetic; encoding untouched.

- [ ] **T41 — Grouped SHAP over emergent groups.** [FABLE A1§4 / A2§6 / A3§2]
  *Depends:* T25 (done), rebuild + retrain. *Done when:* the authoritative SHAP story is
  reported over **emergent** (data-driven, then semantically labeled) feature groups — the
  way indicators are narrated (e.g. elevation + winter temp + SWE + veg type → "alpine
  landscapes") — **not** blanket VIF paring. True redundancies handled case-by-case (the
  closure via T35; any residual bias-proxy individually). Belongs to the results-
  interpretation phase (SCOPE Headline C).

- [ ] **T44 — Contradictory-label ceiling diagnostic.** [FABLE A3§3] *Depends:* rebuild.
  *Done when:* the count of feature-identical / label-disagreeing groups in
  `features_clean.csv` is reported as an irreducible-noise ceiling on separation (context
  for the AUC-PR; pairs with the expected narrow GBM-vs-logistic margin).

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
