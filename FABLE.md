# FABLE.md — feature-strategy review log

Findings on the scientific and machine-learning strategy behind the model's
feature set (not on whether the build *runs* — that is other agents' remit).
Each agent appends its own section below; do not rewrite earlier agents' entries.

---

## Agent #1 — feature science & ML strategy (2026-07-13)

Reviewed: `data/build_feature_table.py`, `data/gee_features.py`,
`data/local_rasters.py`, `data/build_daymet_rasters.py`,
`data/clean_feature_table.py`, `data/build_prediction_data.py`.

Priority tags: **[FIX-BEFORE-RUN]** cheap to fix now, expensive to discover after
the multi-hour build · **[DECIDE]** a design call to make · **[DOCUMENT]** carry
as a stated limitation.

### 1. Train/serve resolution mismatch — [FIX-BEFORE-RUN]
The point table samples features at native scale; the prediction datacube
reprojects **everything to 4 km** (`build_prediction_data.py:43`, `SCALE = 4000`).

| Feature | Trained at | Served at | Severity |
|---|---|---|---|
| Slope, Aspect | 10 m | 4 km | **severe** |
| Elevation | 10 m | 4 km | mild (smooth field) |
| Curvature | 250 m / 1 km | 4 km | moderate |
| Soil (SoilGrids) | 250 m | 4 km | mild |
| Bioclim | 1 km | 4 km | mild |
| FIRMS fire | 4 km | 4 km | consistent ✓ |
| Daymet (SWE/trends) | 1 km raster (nearest) | 1 km raster (nearest) | consistent ✓ |

Slope is the clear problem: a rugged-terrain training point can read 25–30° at
10 m but a few degrees at 4 km. The model learns over a slope range it never
encounters at inference, silently degrading the statewide map. Spatial CV will
NOT catch this — CV is built from the 10 m point samples and never sees the 4 km
distribution. Aspect at 4 km is close to meaningless. Fire (both sides 4 km) and
Daymet (shared 1 km raster) are already consistent; terrain is not.
**Fix:** make the point and datacube paths agree on terrain scale.

### 2. Aspect encoding — [FIX-BEFORE-RUN]
Aspect is carried as raw degrees (0–360), a circular variable: 359° and 1° are
adjacent on the ground but maximally far apart numerically. Decompose into
`northness = cos(aspect)`, `eastness = sin(aspect)`. Also aspect is undefined
where slope ≈ 0 — 3DEP returns a sentinel/arbitrary value on flats that leaks in
as a real number. Cheap at build time, painful to retrofit.

### 3. Ground ice / permafrost feature missing — already plumbed — [FIX-BEFORE-RUN / DECIDE]
Excess ground ice is *the* physical control distinguishing abrupt (thermokarst,
thaw slumps — collapse of ice-rich ground) from gradual (active-layer deepening)
thaw, yet there is **no ground-ice or permafrost feature in the stack**. But:
- `data/Obu2019/` holds the Obu et al. permafrost-probability raster and
  `local_rasters.py` already defines `OBU_TIF` + sampling for it — **never sampled
  into the table** (only used in the module's `__main__` smoke test).
- `data/arctic-permafrost-map/` (Brown/IPA-style) exists and is unused.

Wiring in Obu permafrost probability is low-effort (plumbing exists) and directly
on-target. Highest-value *addition*. Ground ice does not carry the sampling-bias
risk that a surface-water proxy would (see #7).

### 4. Redundancy — [DECIDE / DOCUMENT]
- **19 WorldClim bioclim variables are heavily collinear** (bio01/06/11 ≈
  duplicate temperature; bio12/13/16 ≈ duplicate precip). XGBoost tolerates this
  for prediction, but it splits attribution credit arbitrarily among correlated
  features — directly relevant to a project whose north-star is *identifying*
  indicators. A VIF/correlation paring pass is already noted as README to-do #15;
  flag the SHAP interaction up front.
- **Sand + silt + clay are a closed composition** (sum ≈ 100%): exact redundancy
  by construction, any one determined by the other two. Drop one or use log-ratios.
- **Three overlapping fire features**: ALFRESCO Flammability Index (modeled,
  1900–1999), FIRMS Max Fire Temperature, Fire Detected (observed, 2000+). Not
  fully redundant but be deliberate about carrying all three.
- bio19 (coldest-quarter precip) is effectively a snow proxy overlapping Mean
  Annual SWE.

### 5. One-hot categorical handling — [DECIDE]
One-hots are built from categories present in the training points
(`clean_feature_table.py`, `feats[col].unique()`). (a) Rare Alaska classes
(e.g. Cultivated Crops) become near-constant noise columns — consider collapsing.
(b) A land-cover code present statewide but absent from training points gets no
column and is silently absorbed at inference — verify the datacube category set
is a subset of training's.

### 6. Feature max-fire-temperature choice — [DECIDE]
FIRMS `T21` temporal max is *peak brightness temperature of a detection*, i.e.
how hot a fire burned once, not fire regime. For thaw, occurrence/severity/recency
(burn frequency, time-since-fire, burned fraction) is more mechanistic. The binary
Fire Detected is more defensible than the continuous max.

### 7. Missing features (beyond ground ice) — [DECIDE]
In rough priority: **Yedoma extent** (Strauss et al. — classic ice-rich abrupt
substrate); **topographic wetness index / flow accumulation** (have elevation/
slope/curvature but no hydrological terrain derivative); **NDVI level + trend**
(greening/browning is a strong abrupt-thaw signal). Caution: water/wetness
proximity features may encode the lake-biased *sampling design* rather than a
mechanism (see `diagnostics/feature_provenance.py`). Ground ice and Yedoma avoid
that trap; surface-water proximity does not.

### 8. Temporal provenance — [DOCUMENT]
Feature windows span a century — WorldClim 1970–2000, Daymet trends 1991–2020,
ALFRESCO flammability 1900–1999, veg mode 1950–2008, NLCD 2016, FIRMS 2000+ —
while labels are field observations at various dates. `clean_feature_table.py`
drops `ImageryDates`, so the temporal relationship (does a trend window postdate
the observation it predicts?) can't even be audited. Retain `ImageryDates` as a
non-model column so it stays auditable; note the static-susceptibility framing.

### 9. Coverage risks to pre-flight — [DECIDE] (data-side, but strategic)
- **USGS 3DEP 10 m coverage over Alaska is incomplete** (much is IFSAR-derived or
  absent). If the terrain backbone is NaN over large statewide regions, that's a
  strategy problem, not just a data one.
- **SoilGrids confidence is low at high latitudes**; northern points may be
  largely NaN, making the 12 soil features mostly noise there.

### Notes on what's already sound
Coordinate quarantine (T7), buffered spatial-block CV, no imbalance reweighting
(preserves the sample prior for the likelihood-ratio framing), native NaN routing,
shared `gee_features` definitions across point/datacube paths, and the fire and
Daymet resolution parity are all done well. Critique above is calibrated against
these, not in ignorance of them.

---

## Agent #2 — pipeline methods & objectives review (2026-07-13)

Reviewed: `models/train_xgboost.py`, `models/spatial_cv.py`, `models/predict.py`,
`models/shap_values.py`, `data/clean_feature_table.py`, `data/build_feature_table.py`,
`data/build_prediction_data.py`, `diagnostics/FINDINGS.md`, `TASKS.md`, `SCOPE.md`,
`PIPELINE.md`. Emphasis on evaluation/mapping methods and objective-alignment rather
than the feature set itself (Agent #1's remit); cross-referenced where we overlap.
Same tags: **[FIX-BEFORE-RUN]** · **[DECIDE]** · **[DOCUMENT]**.

### 1. The mapping stage is unfinished and, as-is, contradicts the objectives — [FIX-BEFORE-RUN]
`TASKS.md` T20–T23 and T28 are unchecked, and `models/predict.py` still reflects the
pre-cleanup design. If run tomorrow as written it will:
- emit a **discrete classification** (`prediction_classes.nc`, classification map) at
  `DECISION_THRESHOLD = 0.6` (predict.py:12, 138, 221–227) — SCOPE obj. 1 / PIPELINE
  G19 say the deliverable is the continuous log-evidence surface with **no discrete
  class** and no defensible single threshold;
- ship **no Obu permafrost-domain mask** (T20) and **no Area-of-Applicability layer**
  (T21), though obj. 4 and the "uncertainty as a product" plan depend on the AOA, and
  the map extrapolates well beyond a lake-/road-biased sample;
- gate validity with `min_valid_features_ratio = 0.5` (predict.py:95–97), a near-no-op:
  the many legitimately-zero one-hot columns let almost every pixel clear 50% "valid"
  even where all continuous drivers are missing.
The training run will look successful while the headline map is the wrong product.

### 2. The label is "Non-abrupt," not "Gradual" — [FIX-BEFORE-RUN / DECIDE]
Verified against `Alaska_Permafrost_Thaw_Database_v2.0.0.csv`: `ThawType` ∈
{`Abrupt` 18213, `Non-abrupt` 1327}. `clean_feature_table.py:12` sets class 1 =
*Non-abrupt*, but the glossary, objectives, and the "mode vs. occurrence" positioning
all treat class 1 as **Gradual**. If "Non-abrupt" bundles gradual thaw *and*
stable/no-observed-thaw sites, the target is "abrupt vs. not-abrupt," not thaw *mode* —
which weakens the core novelty claim. One-question check with `webb2026-thawdb`: is
"Non-abrupt" specifically gradual thaw, or everything that isn't abrupt? The code and
the prose currently disagree.

### 3. The 1 km buffer is likely too small for the feature autocorrelation range — [DECIDE]
`spatial_cv.py` fixes `BUFFER_KM = 1.0` at every block size. Prior audit found 62% of
points within 1 km and **97% within 5 km** of a neighbor, and features are gridded at
250 m / 1 km / **4 km**. A 1 km dead zone removes only the nearest tier; 4 km-smoothed
covariates still leak across the seam, so the headline spatial-CV AUC-PR is probably
still optimistic — and the block-*size* sweep doesn't fix it because the buffer stays
1 km even in the large-block "extrapolation" folds. Scale the buffer to the empirical
autocorrelation range, or run a buffer-sensitivity check (1/2/5/10 km) so the reported
number is defensibly stable. `diagnostics/leakage_decay.py` looks like the right tool.

### 4. SHAP explains per-fold refits; the map uses the all-data model — [DOCUMENT]
`shap_values.py` correctly does OOF attribution via per-fold refits (`model.json` is
deliberately unused, shap_values.py:14), while `predict.py` maps the all-data
`model.json`. Both are defensible, but the explained model and the mapped model are
not identical — state this, or a reviewer will read Headline C and Headline A as coming
from one fit when they don't.

### 5. Be ready for XGBoost ≈ penalized logistic under honest CV — [DECIDE]
`FINDINGS.md` had logistic at ~0.805 AP (random split) vs. XGBoost 0.70–0.78
(spatial CV). The rewrite now runs both through *identical* nested folds
(train_xgboost.py:131–137) — the right comparison — but the likely result is a small
GBM margin. Decide the framing now so a narrow gap reads as expected: the case for
GBM + SHAP must rest on nonlinearity/interactions, not on a headline accuracy edge.

### 6. Collinearity paring should precede the SHAP ranking, not follow it — [DECIDE]
Corroborates Agent #1 §4 from the interpretation side. `clean_feature_table.py:100–106`
keeps the full 49-feature set and defers VIF/collinearity to "after." But Headline C is
a *ranked, physically-interpreted* SHAP story, and TreeSHAP splits credit arbitrarily
among correlated features — so the top-of-ranking order is partly a tie-breaking
artifact and will shift when a redundant variable is dropped. For a ranking meant to be
interpreted, pare (or use grouped/clustered SHAP over correlated blocks) *before* the
authoritative SHAP run.

### 7. Feature-build failures are silent until hour ~12 — [FIX-BEFORE-RUN]
(Operational, but it decides whether the strategy above ever gets tested.) Every
`add_feature` in `build_feature_table.py` is wrapped in a bare `try/except` that prints
and continues; a partial failure writes a `features_dirty.csv` with columns missing.
The T30 end-of-run report (lines 312–328) catches raised/all-NaN features but only
after the full build, and a dropped *soil-depth* column will instead `KeyError` in
`clean_feature_table.py`'s depth aggregation. Add a hard required-feature assert before
the CSV is written, and dry-run the GEE track on a few hundred points against the live
project *tonight* — the full run has never completed end-to-end (T30 still open).

### 8. "Prior-free / absolute" log-evidence is a monotonic shift of logit(P) — [DOCUMENT]
`predict.py:133` computes `logit(P_model) − logit(π_sample)` with π_sample read from the
refit data. As a *relative* susceptibility index this is clean; just don't oversell
"absolute," since the subtracted constant is itself the biased sample prevalence.

### On Agent #1 §1 (train/serve terrain scale)
Partial disagreement worth resolving by measurement, not argument. `ee.Terrain.slope`/
`mean_curvature` are computed on the *native-resolution* elevation in both paths, and
the datacube then reprojects the already-computed slope to 4 km — so served slope may
be a subsampled 10 m-slope value rather than a genuine 4 km-slope, making the
distribution shift milder than "severe." But EE `reproject` semantics (recompute vs.
resample at target scale) make this genuinely ambiguous. Broaden T23 (currently scoped
to soil NaN only) into a per-feature training-column-vs-datacube-pixel distribution
comparison so parity is asserted, not assumed.

---

## Agent #3 — feature science & ML strategy (2026-07-13)

Reviewed: `data/gee_features.py`, `data/build_feature_table.py`,
`data/local_rasters.py`, `data/clean_feature_table.py`,
`data/build_prediction_data.py`. Same remit as Agent #1 (feature set, not whether
the build runs) and same tags: **[FIX-BEFORE-RUN]** · **[DECIDE]** · **[DOCUMENT]**.
Agents #1 and #2 already cover most of the terrain-scale, aspect, ground-ice,
collinearity, fire, one-hot, and provenance ground. I reached the same conclusions
independently — treat that as corroboration weight, not new claims — and lead here
with what those entries do **not** contain.

### 1. Datacube double-flips Land Cover & Vegetation Mode — [FIX-BEFORE-RUN] (NEW)
In `build_prediction_data.py`, the categorical one-hots are vertically flipped
**twice** while every other feature is flipped exactly once:
- Land Cover: `landcover_array = np.flipud(sample_local(...))` (line 257), then per
  one-hot `feature_arrays[name] = np.flipud(landcover_data)` (line 263) → net **no
  flip**.
- Vegetation Mode: same pattern at lines 278 and 284 → net **no flip**.
- Contrast Flammability (line 186), a single `np.flipud(sample_local(...))` — the
  correct pattern — and Slope/bioclim/soil, all flipped once.

Net effect: Land Cover and Vegetation Mode layers are **vertically mirrored
relative to the rest of the feature stack**, so on the statewide map the land-cover/
vegetation predictors are spatially inverted against climate, terrain, and soil.
The datacube still builds and produces a plausible-looking surface, which is exactly
why a "does it run" check misses it. This is technically a correctness bug (other
agents' remit) but it silently corrupts the headline map, so flagging here.
Fix: drop the inner `np.flipud` in both categorical loops (lines 263, 284).

### 2. Dimensionality vs. the ~7% minority class — [DECIDE] (NEW angle)
After one-hot expansion the table is ~60–70 columns (19 bioclim + 12 soil + ~28
land-cover/veg one-hots + terrain + climate + fire), while Gradual/Non-abrupt is
~7% of points and feature-space dedup shrinks it further. That is a high
feature-count-to-minority-example ratio → overfitting and unstable SHAP for the very
class the project exists to resolve. The near-zero-variance one-hots (rare Alaska
classes, Agent #1 §5) add columns without signal. The collinearity paring in Agent #1
§4 / Agent #2 §6 is the same fix serving double duty here — cut features and the
p-vs-minority-n ratio improves at the same time.

### 3. Dedup keeps contradictory-label rows at coarse support — [DOCUMENT] (NEW)
`clean_feature_table.py:114` builds the dedup key from all columns except
Latitude/Longitude — which **includes `Class`**. So feature-identical points with
*different* labels are both retained (correctly, they are not duplicates), but at
1–4 km support many nearby points collapse to identical coarse-feature vectors with
conflicting labels. Those contradictory pairs are an irreducible-noise floor no model
can separate; the coarser the feature (bioclim 1 km, FIRMS 4 km), the more of them.
Worth quantifying (count feature-identical / label-disagreeing groups) and stating as
a ceiling on achievable separation, especially alongside Agent #2 §5's "expect a
narrow GBM margin."

### 4. "Fire Detected" is near-constant at 4 km — [DECIDE] (sharpens Agent #1 §6)
Beyond Agent #1's point that FIRMS `T21` max is the wrong fire quantity: the derived
binary Fire Detected = "any detection ever in this ~16 km² cell over the record."
Across the boreal interior that is ≈1 almost everywhere and ≈0 in tundra, so the
feature likely encodes **interior-vs-coastal/tundra geography**, not a thaw mechanism —
a near-constant-within-region column that can proxy the sampling geography. Reinforces
the case for burn frequency / time-since-fire over both FIRMS features.

### 5. Land-cover one-hots encode the sampling design — [DECIDE] (extends Agent #1 §5/§7)
The developed classes (`Developed, Open/Low/Medium/High Intensity`) and `Open Water`
exist in the point set largely *because* collection is road- and lake-biased
(CLAUDE.md; `diagnostics/feature_provenance.py`). These are the categorical analogue
of Agent #1 §7's water-proximity trap: the model can learn road-/lake-proximity as
"signal" and it will surface in the SHAP ranking as a pseudo-indicator. Consider
collapsing the developed/agricultural classes and treating any high "Open Water" or
"Developed" attribution as suspect provenance, not mechanism.

### 6. WorldClim **V1** specifically — [DOCUMENT] (sharpens Agent #1 §8)
The bioclim source is `WORLDCLIM/V1/BIO` (`build_feature_table.py:133`) — the
deprecated V1 (~1960–1990 normals), not V2.1. So the table mixes a 1960–1990 climate
baseline (bioclim) with a 1991–2020 baseline (Daymet trends) in one feature vector.
Either note the two-baseline inconsistency explicitly or move bioclim to V2.1 while the
build is being touched anyway.

### Corroborated independently (no new content — confidence weight only)
Train/serve terrain-scale mismatch (Agent #1 §1 / Agent #2 §205); aspect circularity
(§2); Obu already-plumbed-but-unused + missing ground ice/Yedoma (§3/§7); 19-bioclim
and sand+silt+clay closure collinearity and its threat to the SHAP ranking that is the
north-star deliverable (§4, Agent #2 §6); TWI/NDVI-trend as the highest-value additions
(§7). I concur with all of these on the same reasoning.
