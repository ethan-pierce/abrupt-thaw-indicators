# SCOPE

Scope for the manuscript. **Objectives** are committed goals (the north-star
decomposed). **Brainstorm** is diffuse and explicitly *not vetted* — hunches for
downstream skills to formalize (→ `/derive-equations`) or sharpen into testable
targets (→ `/analyze-system`), not settled claims.

## Key background (the field's current theory this responds to)

Grounded sources live in `REFERENCES.md` and are cited below by key. The
incumbent landscape separates along a three-way distinction the project turns
on (now in the glossary): thaw **stage** (how far thaw has progressed), thaw
**occurrence** (whether a hazard forms), and thaw **mode** (abrupt vs. gradual —
this project's target). No incumbent predicts *mode*.

- **Categorical / rule-based mapping is the oldest incumbent.** Olefeldt et al.
  (2016, *Nat. Commun.*) map circumpolar thermokarst landscapes as discrete
  coverage classes across wetland/lake/hillslope types at regional resolution;
  Jorgenson-style vulnerability mapping is rule-based on ground-ice, thaw rate,
  and terrain. `webb2025-definitions` (226-study review) confirms the field's
  working framework for abrupt thaw is still an explicitly *categorical decision
  tree*, and names AI/data-driven methods as a promising-but-unrealized gap.
- **Continuous ML susceptibility mapping already exists — for *occurrence*, not
  *mode*.** This corrects the earlier framing: "continuous ML beats categorical"
  is *not* novel on its own. `wang2023-arctic-tls` produced a continuous
  random-forest thermokarst-*lake* susceptibility map for the whole Arctic
  (poleward of 60°N); `yang2024-qtp-interpretable` did continuous ML **+ SHAP**
  thermokarst-hazard susceptibility on the Qinghai–Tibet Plateau. Both predict
  *where one landform occurs*, one model per hazard — not which *mode* thaw takes.
- **The closest ML neighbors map stage or detect landforms, not mode.**
  `zhang2025-thawstage` maps four post-fire thaw *stages* in interior Alaska via
  repeat-LiDAR change detection over 2500 km² (~79% acc., fire-controlled);
  `yang2023-rts-cnn` detects discrete retrogressive thaw slumps pan-Arctic. Both
  are informative and neither answers "abrupt or gradual, and why."
- **The target question is nearly unasked — and the nearest prior answer is
  in-house.** `webb2026-thawdb` (this repo's Thaw Database; user is 2nd author)
  already related abrupt-vs-non-abrupt to slope, elevation, and solar radiation —
  a *bivariate, descriptive* answer. Nobody has done the **multivariate, ranked,
  predictive, per-point, statewide-spatialized** attribution of thaw mode.
- **Abrupt thaw is disproportionately important but coarsely resolved.** It
  affects a small land fraction yet drives an outsized share of the
  permafrost-carbon feedback (Turetsky et al. 2020 — not yet grounded). Its
  controls are described mechanistically, not ranked quantitatively as
  observable predictors.
- **Known data caveat (now quantified).** `webb2026-thawdb` is lake-dominated:
  10,625 of 18,213 abrupt points are thermokarst-lake features, sampling is
  denser along roads, and the authors themselves warn this "could skew model
  outputs toward aquatic thaw processes while underrepresenting terrestrial
  forms." The 93.2%/6.8% abrupt/non-abrupt split matches this repo's balance.
- **Data lineage — RESOLVED.** This repo's Thaw Database *is* `webb2026-thawdb`
  (confirmed by the user). A data-version/model-rerun reconciliation to match the
  published v2.0.0 exactly is a separate, deferred issue.

## Objectives (north-star decomposed — target question: *"why is this point undergoing abrupt rather than gradual thaw?"*)

The contribution is **thaw *mode* + its explanation**, not a continuous map per
se (already done for *occurrence* — see Key background). Two headlines:

1. **Headline A — classify thaw mode.** Predict abrupt-vs-gradual thaw *mode*
   from geospatial features at any point, spatialized as a continuous statewide
   (Alaska) probability surface. Novel target: incumbents predict stage or
   single-hazard occurrence, none predict mode.
2. **Mechanism B (what makes A possible) — one mechanism-agnostic model.** A
   single classifier over a shared feature stack fuses lake/wetland/hillslope
   thaw forms, rather than one model per landform. The Thaw Database's breadth
   (`webb2026-thawdb`) is the enabling asset; its lake-dominance is the honest
   limit (scopes A and C).
3. **Headline C (secondary) — explain the mode.** Rank and physically interpret
   the geospatial indicators (SHAP) that drive the abrupt-vs-gradual call — the
   *why* half of the target question, and the part no incumbent (including the
   bivariate `webb2026-thawdb` analysis) has done multivariately and per-point.
   Caveat to carry: SHAP attributions inherit the DB's lake-dominance and may
   encode "near a lake" as proxy for mechanism. → `/analyze-system`
4. **Establish predictive credibility.** Calibrated probabilities and honest
   performance under the ~93/7 class imbalance, with demonstrated generalization
   (CV stability, train/test gap). See Open items — the leakage check gates this.
5. **Position against the incumbents.** Not "continuous vs. categorical" (that
   fight is settled) but "mode vs. occurrence/stage" and "explained vs. mapped":
   what predicting *which pathway* — and *why* — buys over Olefeldt categorical
   classes, `wang2023`/`yang2024` occurrence-susceptibility, and `zhang2025`
   stage maps.

## Brainstorm (NOT VETTED — diffuse directions)

- **Mode-vs-incumbent head-to-head** — overlay the mode-probability surface on
  Olefeldt classes *and* on occurrence-susceptibility (`wang2023`); quantify where
  predicting *mode* diverges from predicting *occurrence*. (No longer the headline
  novelty — now a validation/positioning move for objective 5.)
  *What would make this real:* a defined comparison metric/region. → `/analyze-system`
- **Terrestrial-vs-aquatic sampling bias** — check whether the training DB is
  lake-dominated and scope the map's claims (and coverage) accordingly.
  *What would make this real:* a breakdown of DB points by thaw form.
- **Indicators as mechanism** — curvature / SWE trend / silt / nitrogen read as
  proxies for ground-ice, drainage, and soil texture; could crystallize into a
  physically-motivated susceptibility index. → `/derive-equations`
- **Missing driver: fire — PARTLY RESOLVED.** `zhang2025-thawstage` found fire
  dominant for thaw stage; fire is *not* absent here — `build_feature_table.py`
  pulls ALFRESCO flammability and FIRMS max-fire-temperature, and
  `clean_feature_table.py` keeps Maximum Fire Temperature. Open question is
  whether the *representation* (instantaneous max temp / flammability vs. fire
  *history*/time-since-fire) is adequate, not whether fire is present.
- **Uncertainty as a product** — propagate calibrated probability into a
  mode-with-uncertainty map, not just a point estimate. → `/analyze-system`
- **Transferability** — does an Alaska-trained mode classifier extend pan-Arctic?
  (Fallback framing if "mode is the story" ever needs a broader-reach angle.)

## Open items (route, don't resolve here)

- **Class encoding — RESOLVED (was doc-only drift).** An `/audit-repo` pass
  confirmed the live pipeline *and* the calibrated track are consistent:
  `0 = Abrupt` (majority ~93%), `1 = Gradual` (minority ~7%), set in
  `clean_feature_table.py`. Earlier "blocker" framing was wrong — it trusted a
  stale `CLASS_ENCODING_VERIFICATION.md` that actually described a legacy
  `Abrupt = 1` model. Confusion matrix / SHAP plots from the live scripts are
  labeled correctly. Only legacy artifacts used the reverse (removed/archived).
- **Canonical model + map robustness — needs a science-skill investigation, not
  cleanup.** `model.json` (300 trees, `scale_pos_weight≈12.7`, uncalibrated) and
  the calibrated family (`model_calibrated.pkl`, 737-tree base, no reweighting,
  sigmoid-calibrated to the ~93%-abrupt prior) are two distinct models on the same
  49 features/encoding. A head-to-head comparison found they **disagree
  substantially on the statewide map**: Pearson r ≈ 0.66, **~37% of valid pixels
  flip class at threshold 0.6**, mean P(Abrupt) 0.47 (uncal) vs 0.80 (calibrated).
  The divergence is driven by **class-imbalance handling** (reweighting vs
  calibrating to a prior that reflects the DB's ~93%-abrupt, lake-dominated
  sampling bias). Choosing a canonical model therefore changes the headline map
  and interacts with the sampling bias — a scientific call. Operative model
  remains `model.json` (unchanged) pending this. → `/analyze-system` (imbalance /
  sampling bias / map robustness), `/verify-code` (implementation).
- **Near-perfect discrimination — possible leakage/overfit.** Both models score
  AUC-ROC ≈ 0.99 and AUC-PR ≈ 0.9999 (93% prevalence). Warrants a leakage /
  feature-independence check before the numbers go in the manuscript. → `/verify-code`
- **Metric "positive class" = Gradual** — `train_xgboost*.py` report AUC-PR /
  precision / recall / F1 for class `1` (Gradual, minority) via `predict_proba[:, 1]`,
  while `predict.py` maps P(Abrupt) = `predict_proba[:, 0]`. Coherent, but the
  write-up must state which class each headline metric describes. → `/verify-code`
