# SCOPE

Scope for the manuscript. **Objectives** are committed goals (the north-star
decomposed). **Brainstorm** is diffuse and explicitly *not vetted* — hunches for
downstream skills to formalize (→ `/derive-equations`) or sharpen into testable
targets (→ `/analyze-system`), not settled claims.

## Key background (the field's current theory this responds to)

- **Where abrupt thaw can occur is mapped categorically and by rule.** Olefeldt
  et al. (2016, *Nat. Commun.*) map circumpolar thermokarst landscapes as
  discrete coverage classes (Very High → None) across wetland/lake/hillslope
  types, at regional resolution. Jorgenson-style vulnerability mapping in central
  Alaska is rule-based on ground-ice, thaw rate, and terrain.
- **Closest ML neighbor maps stage, not mode.** Zhang et al. map permafrost
  *thaw stages* in interior Alaska from ML + remote sensing (~79% accuracy,
  largely fire-controlled) — informative, but it does not distinguish *abrupt vs.
  gradual* thaw mode as a continuous probability.
- **Abrupt thaw is disproportionately important but coarsely resolved.** It
  affects a small land fraction yet is understood to drive an outsized share of
  the permafrost-carbon feedback (Turetsky et al. 2020). Its controls are
  described mechanistically, not ranked quantitatively as observable predictors.
- **Known data caveat:** abrupt-thaw databases are lake/aquatic-dominated and
  under-sample terrestrial thaw — a sampling bias the paper must scope against.
- **Likely data lineage:** the "comprehensive database of thawing permafrost
  locations across Alaska" (ESSD, 2026, v2.0.0) is probable kin to this repo's
  Thaw Database; align citation and version.

## Objectives (north-star decomposed — "the map is the story")

1. **Primary — produce the map.** A continuous, statewide (Alaska) probability
   surface for abrupt-vs-gradual thaw, predicted from geospatial features, that
   improves on categorical landscape-class susceptibility.
2. **Establish predictive credibility.** Calibrated probabilities and honest
   performance under the ~94/6 class imbalance, with demonstrated generalization
   (CV stability, train/test gap).
3. **Explain the map.** Rank and physically interpret the geospatial indicators
   (SHAP) that drive the predictions — the map's supporting evidence.
4. **Position against the incumbent.** Show what a continuous, data-driven
   surface buys over Olefeldt-style categorical classes where they overlap.

## Brainstorm (NOT VETTED — diffuse directions)

- **Continuous vs. categorical head-to-head** — overlay the probability surface
  on Olefeldt classes; quantify agreement and where continuity adds signal.
  *What would make this real:* a defined comparison metric/region. → `/analyze-system`
- **Terrestrial-vs-aquatic sampling bias** — check whether the training DB is
  lake-dominated and scope the map's claims (and coverage) accordingly.
  *What would make this real:* a breakdown of DB points by thaw form.
- **Indicators as mechanism** — curvature / SWE trend / silt / nitrogen read as
  proxies for ground-ice, drainage, and soil texture; could crystallize into a
  physically-motivated susceptibility index. → `/derive-equations`
- **Missing driver: fire** — Zhang et al. found fire dominant for thaw stage;
  is fire history an absent feature that would sharpen the map?
- **Uncertainty as a product** — propagate calibrated probability into a
  susceptibility-with-uncertainty map, not just a point estimate. → `/analyze-system`
- **Transferability** — does an Alaska-trained model extend pan-Arctic? (This is
  the fallback framing if "method is the story" ever eclipses "map is the story.")

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
