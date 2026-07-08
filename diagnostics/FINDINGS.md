# verify-ml findings — abrupt-vs-gradual thaw classifier

**Stamp (reproducible only against all three):**
- Code: git `347c9cd` (working tree: `diagnostics/` untracked, no pipeline edits)
- Data: `features_clean.csv` sha256 `99E52C17FDC606D0…` (19,272 rows × 49 features + Class);
  reconstructed with coords from `features_dirty.csv` sha256 `3F10A95C6C151109…`
- Model checked: representative grid config refit in-probe (not the committed
  `model.json`, sha256 `AC75D4B81D4498CA…`, which is the pre-retrain artifact)
- Seeds: split/model `42` (pipeline draws split_seed=8 from `default_rng(42)`); shuffle `1000–1004`
- Run date: 2026-07-08

**Scope caveat.** The retrain (README to-do #6) has **not** run: `features_clean.csv`
is still the *old pared 49-feature set* and `model.json` is the old model. This audit
targets the **protocol** — split, dedup, baselining, metric choice, feature provenance —
which the pipeline code applies unchanged across the retrain, so every flag below
will persist into the new numbers unless the protocol itself changes. The absolute
AUC values will move; the *mechanisms* will not. Re-run this suite after each retrain.

---

## What was established to hold

- **No broken-pipeline / target-passthrough leakage.** Shuffle-label probe (permute
  train labels, refit, score real test) collapses to chance: AUC-ROC 0.48±0.05,
  AUC-PR 0.064 (floor 0.057). Real predictive signal exists; the score is not an
  index bug or a target column fed through a side channel. — `baseline_and_shuffle.py`
- **The floor is real and the trivial baselines behave.** Majority/prior/stratified
  dummies sit exactly at chance (AUC-ROC 0.50, AUC-PR 0.057). — `baseline_and_shuffle.py`
- **Coordinates are not in the model.** Confirmed by reconstruction: `features_clean.csv`
  carries none of Latitude/Longitude (README #2 corroborated).

## Flags raised

### 1. Random split inflates the imbalance-appropriate metric — spatial leakage (CONFIRMED)
`spatial_leakage.py` (B). Under the pipeline's random split the metric is
AUC-PR 0.898±0.018; under spatial block-holdout (GroupKFold on 0.5–2° grid cells)
it falls to **0.70–0.78 with 4–6× the fold-to-fold variance** (sd up to 0.115).
AUC-ROC barely moves (0.993→0.97–0.98) because ROC is insensitive to the minority —
which is exactly why the headline should not lean on it. The honest
generalization-to-new-locations number is ~0.70–0.78 AUC-PR, materially below the
random-split 0.90 and far below any 0.9999 claim. **The manuscript's performance
claim must be measured under spatial CV, not a random split.**

### 2. Sampling geometry drives the inflation (CONFIRMED)
`spatial_leakage.py` (A). Exact/near-exact feature twins across the split are rare
(≤0.3% of test rows), so it is *not* literal duplicate rows. It is spatial
interleaving: **40% of test points lie within 0.5 km of a train point, 62% within
1 km, 97% within 5 km.** Clustered (lake/road-biased) sampling + coarse gridded
features means the random split trains and tests on the same neighbourhoods.
Exact-only `drop_duplicates` in `clean_feature_table.py` does not address this.

### 3. The model barely beats a simple baseline once measured honestly (CONFIRMED)
`baseline_and_shuffle.py`. Plain logistic regression (balanced) scores AUC-ROC 0.981,
**AUC-PR 0.805** on the random split — at or above XGBoost's *spatial-CV* AUC-PR
(0.70–0.78). A depth-1 decision stump scores AUC-ROC 0.85. No fitted baseline is
present anywhere in the pipeline (only a drawn "chance level" line). **The floor a
gradient-boosted model must clear is logistic regression, not chance** — and on the
honest split it clears it by little. State this comparison in the manuscript.

### 4. Discrimination rides on smooth gridded covariates that can proxy location (PLAUSIBLE)
`feature_provenance.py`. Top univariate separators are soil chemistry and climate,
not water/lake one-hots as the lake-bias story would predict: Nitrogen 0–30 cm
(AUC-PR 0.67), Precipitation Seasonality (0.56), Soil Organic Carbon (0.55), Silt,
Sand, Flammability. These are spatially smooth products; combined with flag 2, they
can behave as low-frequency spatial coordinates rather than thaw mechanisms. This is
the load-bearing risk for **Headline C (SHAP explanation)**: the story will center on
soil chemistry, which must be defended as mechanism vs. spatial-smoothness proxy
(e.g. does the soil-feature signal survive spatial CV / partialling out region?).

### 5. Soil features are 16% missing; missingness is unaudited (PLAUSIBLE)
The top separators (Nitrogen, SOC, texture) have ~83.6% coverage. XGBoost routes NaN
to a learned default direction, so if missingness correlates with class it becomes a
free signal. Not yet tested. → check whether NaN-ness of soil features is predictive.

### 6. Evaluation-code hygiene (reporting, not validity) (CONFIRMED)
- `train_xgboost.py:213–221` extracts `split0..4` from a **10-fold** grid (misses
  splits 5–9) and prints them as "CV F1" though the grid `scoring='neg_brier_score'` —
  the printed "CV F1" is mislabeled negative-Brier.
- Three scripts use **three different, unpersisted splits**: `train_xgboost.py`
  (`default_rng(42)`→seed 8), `train_xgboost_calibrated.py` (`random_state=42`),
  `shap_values.py` (`default_rng(100)`). SHAP therefore explains a test set ~70% of
  whose rows were in the model's *training* set (README #9). No single canonical,
  saved train/test index exists — required for a result keyed to code+data+seed.
- `accuracy` is reported prominently though it is meaningless at 94% prevalence
  (majority vote scores 0.94). Fine only if never headlined.

## Recommended protocol changes (for `/develop-model` — not applied here)
1. Adopt **spatial block CV** (GroupKFold on grid cells, or leave-region-out) as the
   reporting protocol; headline the spatial-CV AUC-PR with its across-fold spread.
2. **Persist one canonical train/test/val split** (saved indices) and share it across
   train, calibrate, and SHAP.
3. Add **logistic regression as the reported baseline** the model must beat.
4. Headline **AUC-PR (minority=Gradual)**, not AUC-ROC or accuracy; report the chance
   floor (=prevalence) alongside.
5. Audit **soil-feature missingness** and the mechanism-vs-spatial-proxy question for
   the SHAP story before Headline C is written.

## Suite (re-runnable)
- `_data.py` — reconstructs the model input from `features_dirty.csv` with coords,
  asserts row-for-row match to `features_clean.csv`.
- `baseline_and_shuffle.py` — Move 1 floor (dummies, logistic, stump) + Move 2 shuffle-label.
- `spatial_leakage.py` — near-duplicate census + random-CV vs spatial-block-CV.
- `feature_provenance.py` — univariate separating power per feature; proxy flags.
