# verify-ml findings — abrupt-vs-non-abrupt thaw classifier

Regenerated against the **rebuilt** pipeline (T28). The prior audit targeted the
pre-retrain 49-feature / random-split protocol; this one is stamped to the current
70-feature model, the spatial-block-CV reporting protocol, and the new closeout
invariants. Class encoding as always: **`0 = Abrupt` (majority), `1 = Non-abrupt`
(minority, the AUC-PR "positive")**.

**Stamp (reproducible against all of these):**
- Diagnostics code: git `338a253` (clean tree), run 2026-07-16.
- Pipeline artifacts: produced at manifest git `5b250ae` (dirty at retrain).
- Data: `features_clean.csv` sha256 `faf00fecd366c5de…` (19,288 rows × 70 model
  features + `Class` + Lat/Lon metadata); reconstructed with coords from
  `features_dirty.csv` sha256 `20409ec9edeab4fe…`.
- Model: `models/model.json` sha256 `6629bc612abf762b…` (operative all-data refit;
  OOF probes refit per fold and do **not** load it).
- CV protocol: `albers_grid`, operative **10 km** blocks, **buffer 0.0 km**, 5 outer /
  5 inner folds, seeds `SPLIT=MODEL=CV=42` (`train_xgboost.py` draws random split
  seed 8 from `default_rng(42)`); shuffle seeds 1000–1004.
- Positive-class (Non-abrupt) prevalence **0.0574** — the AUC-PR chance floor.

**Scope.** This suite audits the **protocol** — coordinate quarantine, split geometry,
buffering, baselining, held-out attribution, train/serve construction, and the
label-noise ceiling — not the manuscript's scientific claims. Absolute AUC values
move with each retrain; the *mechanisms* below are properties of the pipeline code and
persist. Re-run the whole suite after any retrain or feature-set change.

---

## Invariants that HOLD

- **Coordinates are quarantined from the model.** `_data.py` reconstructs the exact
  model input step-for-step from `features_dirty.csv` and asserts row-for-row equality
  to `features_clean.csv`; Latitude/Longitude are carried only as **metadata**, never
  inside `X` (`X=(19288, 70)`, coords present for every row but excluded and verified
  separately). The spatial probes can therefore group by location without the model
  ever seeing it. — `_data.py`

- **No target-passthrough / broken-pipeline leakage.** Shuffle-label probe (permute
  train labels, refit, score the real test set) collapses to chance: AUC-ROC
  **0.493 ± 0.022**, AUC-PR **0.062 ± 0.008** (floor 0.0574). The signal is real, not an
  index bug or a target fed through a side channel. — `baseline_and_shuffle.py`

- **The trivial baselines sit exactly at chance.** Majority / prior / stratified dummies
  score AUC-ROC 0.50, AUC-PR 0.057–0.058. — `baseline_and_shuffle.py`

- **Held-out SHAP is genuinely out-of-fold.** `models/shap_values.py` refits the
  operative hyperparameters on each fold's train subset and runs TreeSHAP on the
  **held-out** points only, pooling across folds so every attribution comes from a model
  that never trained on that point; the all-data `model.json` is deliberately not used.
  Grouped-family SHAP outputs (T41) in `output/` are current. — `shap_values.py`,
  `shap_groups.py`

- **Train/serve construction parity: PASS.** The parity gate compares each training
  column against the datacube pixel at matched 1 km cells: **70 features, identical set
  and order**; **60 clean / 10 offset-sensitive / 0 genuine construction flags**. The 10
  offset-sensitive features are spatially singular / small-patch (Upstream Area, HND,
  curvature, Northness/Eastness, Open Water) whose matched parity rises toward the cell
  centre — sub-cell geometry, not a unit/transform slip. Soil is native-sampled at 250 m
  on both paths (the old "reproject-average" concern is stale); the MODIS >70°N fire QA
  gap reproduces (~4.8% serve NaN). Two categorical classes are silently absorbed into
  the reference bucket — Moss (1,097 cells) and Barren lichen moss (18 cells), negligible
  area. — `train_serve_parity.py` → `train_serve_parity.md`

- **Hard label noise does not bound separation.** Only **4** feature-identical /
  label-disagreeing pairs exist (8 rows, 0.04%; 4 of the 1,107 minority points), all
  within ~0.002 km — the same source pixel. Irreducible ceilings: accuracy 0.9998,
  oracle AUC-PR ≈ 1.0. So the gap the GBM leaves to a perfect score, and the narrow
  GBM-vs-logistic margin, are **not** explained by contradictory labels; they come from
  *soft* feature-space overlap between opposing-label points. — `contradictory_labels.py`

## Flags that persist (the honest-generalization story)

### 1. Random split inflates the minority metric — spatial leakage (CONFIRMED)
Under the pipeline's own random split, XGBoost scores AUC-PR **0.904** (5-fold random
StratifiedKFold **0.908 ± 0.011**). Under spatial block-holdout it falls to
**~0.73–0.79** (GroupKFold 0.5°/1°/2° → 0.786 / 0.734 / 0.757, up to ~13× the fold-to-fold
variance). The production sweep (`output/cv_sweep_results.json`) agrees: pooled-OOF AUC-PR
by block size is 0.82 (10 km) / 0.84 (25 km) / 0.78 (50 km) / 0.81 (100 km) / 0.77
(200 km), across-fold σ 0.07–0.12; operative-10 km selection AUC-PR **0.843**. AUC-ROC
barely moves (0.994 → 0.98) because ROC is minority-insensitive — which is why the
headline must be **spatial-CV AUC-PR with its across-fold spread**, not ROC, not the
random-split number. — `spatial_leakage.py`, `repeated_cv.py`, `output/cv_sweep_results.json`

### 2. Sampling geometry drives the inflation (CONFIRMED)
Literal feature-twins across the random split are rare (0.1% of test rows at feature
distance ≤ 0.01). The leakage is **geographic interleaving**: **19% of test points lie
within 0.1 km of a train point, 41% within 0.5 km, 62% within 1 km, 97% within 5 km.**
Clustered (lake/road-biased) sampling + coarse gridded features means a random split
trains and tests on the same neighbourhoods. Exact-only `drop_duplicates` cannot address
this — block holdout can. — `spatial_leakage.py`

### 3. The margin over a linear baseline is real but modest (CONFIRMED)
On the random split, balanced logistic regression scores AUC-PR **0.834** — near
XGBoost's *spatial-CV* range. The honest comparison is on matched spatial folds: across
20 block→fold reshuffles at 10 km, XGBoost **0.852 ± 0.011** vs logistic **0.776 ± 0.017**,
margin **+0.076 ± 0.019** — outside 2σ of partition noise, so the gradient-boosted edge
is a **stable signal**, not partition luck, but small. The floor a GBM must clear is
logistic regression, not chance; state this in the manuscript. — `baseline_and_shuffle.py`,
`repeated_cv.py`

### 4. Buffer choice is empirical, and block holdout (not a metric buffer) is the tool (CONFIRMED)
On the random split, the leakage-specific gap (targeted near-twin removal vs a
matched-count random-removal control, gap > 0.02) is present and **contiguous through
~2 km**, then collapses into data-depletion noise once >90% of the pool is stripped
(86.5% removed at r = 2 km). A dispersed random test set cannot sustain a large clean
buffer — the argument for **block** holdout. On the production block splitter the buffer
sweep shows no targeted-vs-control gap above 0.02 *on top of* the block folds, so a metric
buffer would mainly strip the sparse minority without removing genuine leakage: the
committed operative choice is **block CV with `buffer_km = 0.0`** (`cv_config.json`, T43).
— `leakage_decay.py`, `block_cv.py` (→ `block_buffer_decay.png`)

### 5. Discrimination rides on spatially smooth covariates that can proxy location (PLAUSIBLE)
Top univariate separators are now **climate and soil**, not water/lake one-hots: Annual
Mean Temperature (AUC-PR 0.69), Nitrogen 30–200 / 0–30 cm (0.66), Mean Temp of Warmest /
Wettest Quarter (0.62), Precip of Coldest Quarter, Trend in precipitation, Precipitation
Seasonality, Soil Organic Carbon (0.55). These are smooth gridded products; combined with
flag 2 they can behave as low-frequency spatial coordinates rather than thaw mechanisms.
This is the load-bearing risk for the **SHAP interpretation headline**: the story centres
on climate/soil, which must be defended as mechanism vs. spatial-smoothness proxy (does
the signal survive spatial CV / partialling out region?). — `feature_provenance.py`

### 6. Soil features are ~16% missing; missingness is unaudited for signal (PLAUSIBLE)
The soil separators (Nitrogen, SOC, texture) have ~83.6% train coverage; XGBoost routes
NaN to a learned default direction, so if missingness correlates with class it becomes
free signal. Serve-side soil NaN is lower (5.1–5.2% statewide) — the train/serve NaN-rate
gap is itself a representativeness signal (parity gate). Not yet tested as a predictor.

### 7. The map's reliability is a representativeness question, layered separately (CONFIRMED)
The parity gate quantifies the lake-/road-collection bias directly: training points sit
in systematically flatter, wetter, lower-drainage, more-open-water locations than the
statewide grid (Slope 0.74° vs 3.92°, HND 1 m vs 17 m, Open Water active 0.43 vs 0.04).
This is why sample prevalence ≠ landscape prevalence, why calibrated probabilities are
not defensible, and why reliability is a **separate** layer: the Area-of-Applicability.
The AOA uses a rank-CDF SHAP-weighted dissimilarity index; OOF AUC-PR stayed ~15× the
prevalence floor across the entire *sampled* DI range (no decay), so the threshold
(DI = 0.506) is set at the edge of the measured-skill envelope — cells beyond it are
flagged as genuine extrapolation, not scored. Spearman(DI, |residual|) = 0.489.
Leave-region-out (case B, 3 km buffer), scoring the **operative model** (spw=1, selected
hparams — identical to the Fig 5a curve), degrades gracefully rather than collapsing:
AUC-PR ~0.84 at ~17 km reach (50 regions) down to **0.54 at ~251 km** median distance-to-train
(3 regions) — still ~9× the 0.057 floor.
— `aoa_calibration.py` → `models/aoa_threshold.json`; `extrapolation_range.py`
(→ `extrapolation_range.png`).

## Resolved since the prior audit
The old audit's flag 6 (evaluation hygiene) is addressed by the rebuild:
- One **canonical split/seed is persisted** (`cv_config.json` / `run_manifest.json`,
  `SPLIT_SEED=42`) and shared across train, sweep, and OOF SHAP — the "three different
  unpersisted splits" issue is gone; OOF SHAP now explains points held out of their own fold.
- Reporting **headlines spatial-CV AUC-PR** with across-fold spread and the prevalence
  floor; **accuracy is not reported** (PIPELINE.md), removing the misleading 94%-prevalence
  accuracy and the mislabeled "CV F1".

## Suite (re-runnable) — mapped to the invariants
- `_data.py` — reconstructs the model input from `features_dirty.csv`, asserts equality to
  `features_clean.csv`, carries coords as metadata. **[coord quarantine]**
- `baseline_and_shuffle.py` — dummy / logistic / stump floor + shuffle-label probe.
  **[baselines, no-passthrough]**
- `spatial_leakage.py` — near-twin census + random-CV vs spatial-block-CV.
  **[spatial leakage]**
- `repeated_cv.py` (→ `repeated_cv.png`) — partition-noise spread of the AUC-PR-vs-scale
  curve + XGBoost-vs-logistic margin stability. **[spatial leakage / baselines]**
- `leakage_decay.py` (→ `leakage_decay.png`), `block_cv.py` (→ `block_buffer_decay.png`) —
  empirical buffer sizing on the random split and the production block splitter.
  **[empirical buffer]**
- `contradictory_labels.py` — feature-identical / label-disagreeing ceiling.
  **[contradictory-label ceiling]**
- `feature_provenance.py` — per-feature univariate separating power; proxy flags.
- `train_serve_parity.py` (→ `train_serve_parity.md/.png`) — per-feature train/serve
  construction gate. **[parity gate]**
- `aoa_calibration.py` (→ `aoa_calibration.png`, `models/aoa_threshold.json`),
  `extrapolation_range.py` (→ `extrapolation_range.png`) — reliability / applicability.
- `probe_native_sampler.py`, `probe_native_serve.py`, `probe_terrain_scale.py` —
  Earth-Engine serve-path sampling probes (require EE auth; support the parity gate).

_OOF SHAP itself lives in the model pipeline (`models/shap_values.py`), exercised here via
`aoa_calibration.py`._
