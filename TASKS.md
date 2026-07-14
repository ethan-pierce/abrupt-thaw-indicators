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

- [ ] **T37 — Terrain train/serve scale: probe first, don't coarsen blind.** [FABLE A1§1 / A2]
  The point path samples slope/curvature at native scale (`build_feature_table.py:104`,
  `reduceRegion(mean, scale=10)`); the datacube extracts at `SCALE=4000`
  (`build_prediction_data.py:134`). Whether this is a *severe* or *mild* mismatch
  hinges on EE `reproject` semantics (recompute-at-4 km vs. resample-a-10 m-value),
  which is unknown. *Depends:* — *Done when:* a quick GEE probe (~200 points, native-mean
  vs. 4 km-reproject for slope + both curvature scales) settles recompute-vs-resample;
  **if** divergence is material, the coarsen-vs-matching-columns fix is decided and
  applied to both paths **before** the full build; if mild, documented and left. Aspect
  is already reprojection-safe via T32. *(The per-feature parity confirmation is the
  DEFERRED T23.)*

- [ ] **T35 — Systematic feature-transform audit.** [user-requested; generalizes A1§2]
  For every feature decide whether it needs a transform, in three buckets: **(1)
  non-monotonic-for-correctness** (circular → cos/sin; signed quantities); **(2)
  must-precede-4 km-reprojection-averaging** (heavy-tailed → `log` *before* the mean —
  the datacube averages, a non-tree op); **(3)** linear-baseline (T13) / SHAP-readability
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

- [ ] **T32 — Aspect → northness/eastness.** [FABLE A1§2] Both paths. Replace raw `Aspect`
  (degrees, circular; `build_feature_table.py:110-111`, `build_prediction_data.py:140-143`)
  with `northness = cos(aspect)`, `eastness = sin(aspect)`. On flats (**slope < 1°**) set
  both to `0` (no preferred direction — keeps the row's other terrain info). Drop raw
  `Aspect` from the model set entirely. *Depends:* — *Done when:* both paths emit
  northness/eastness, flats are neutralized, and no raw `Aspect` column remains.

- [ ] **T33 — Add Yedoma as a feature.** [FABLE A1§7] The excess-ice/ground-ice control
  that mechanistically separates abrupt from non-abrupt thaw (replaces the rejected
  Obu-as-feature — Obu stays mask-only, T20). LOCAL track. Source **in-repo**:
  `data/IRYP_v2_yedoma_confidence_Shapefile/IRYP_v2_yedoma_confidence.shp` (IRYP v2,
  Strauss et al.; **EPSG:3571** — reproject points, as with other local sources). Carries
  a `confidence` class (`conf_id`), so sample either as **binary presence** (in any
  polygon / not) or as **ordinal confidence** (0 = none → confirmed); points outside all
  polygons = 0. *Depends:* — *Done when:* point-in-polygon (points) + rasterize-to-grid
  (datacube) emit a Yedoma feature in both paths. *(The point-in-polygon machinery here
  also makes Brown ground-ice `CONTENT` a cheap future add.)*

- [ ] **T34 — Add hydrological terrain features from MERIT Hydro.** [FABLE A1§7]
  GEE track, `MERIT/Hydro/v1_0_1` (official catalog — **not** the `sat-io` community
  layer). Add `hnd` (height above nearest drainage, raw) and `log(upa)` (for the water-
  convergence signal). *Depends:* — *Done when:* both paths emit `hnd` and `log(upa)`.
  Document the 4 km-reprojection caveat on `log(upa)` (heavy-tailed area averaged to
  4 km; the single worst feature for the T37 scale question — flag it there too).

- [ ] **T36 — Fix fire representation (Package B).** [FABLE A1§6 / A3§4]
  **Drop** continuous `Maximum Fire Temperature` (peak brightness of one detection, not
  regime) and binary `Fire Detected` (near-constant at 4 km → interior-vs-tundra
  geography) — this **reverts the completed T4/T18**. **Keep** Flammability Index
  (long-term modeled regime proxy). **Add** MODIS `MCD64A1`-derived **time-since-last-fire**
  + **burn-count**, materialized to a local raster via the `build_daymet_rasters.py`
  pattern, kept **near native ~500 m** (datacube resamples to 4 km later). *Depends:* — *Done
  when:* both paths carry Flammability + time-since-fire + burn-count and no FIRMS
  `T21`/`Fire Detected` columns; the **24-yr right-censoring** caveat ("no fire since
  2000" ≠ never-burned) is documented.

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
  and features up to 4 km, near-seam leakage may inflate the headline AUC-PR. *Depends:*
  rebuild. *Done when:* `diagnostics/leakage_decay.py` measures the leakage-decay range
  (where OOF performance stops dropping as the buffer grows), the operative buffer is set
  to that range with **no nominal-scale floor** (let the data decide — 4 km is a
  *resampling* grid, not native; coarsest native ≈ 1 km), and a buffer-sensitivity sweep
  (1/2/5/10 km) is reported. *(Not the explanation for the old random-split near-perfect
  discrimination — that predates the buffer; separate leakage question → SCOPE.)*

- [ ] **T23 — Train/serve parity gate (broadened).** [FABLE A2 / A1§5 / A3§5; absorbs T42 + the T37 tail]
  *Depends:* rebuild. *Done when:* a per-feature **training-column-vs-datacube-pixel
  distribution-parity** check is documented for **every** feature (not soil-NaN only),
  including: soil-NaN reproduction; terrain slope/curvature (confirming the T37 probe's
  expectation); and the **land-cover/veg category-set subset check** (report any class
  present statewide but absent from training points, and the area affected — silent
  reference-bucket absorption). No change to one-hot construction.

- [ ] **T20 — Obu domain mask.** [G17] Soft-mask/weight by Obu PerProb
  (`data/Obu2019/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif`; `local_rasters.OBU_TIF` /
  `sample_points`), resampled to the 4 km Albers grid; replaces the feature-validity keep
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
