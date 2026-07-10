# abrupt-thaw-indicators

## Project identity

A machine-learning research project in permafrost geoscience. It uses geospatial
features — drawn from remote sensing, community data products, and climate
reanalysis — to distinguish **abrupt** from **gradual** permafrost thaw across
Alaska. The core method is a gradient-boosted (XGBoost) classifier interpreted
with SHAP values, applied both to labeled field points and to a gridded
statewide feature stack.

## North-star

Identify the geospatial indicators that distinguish abrupt from gradual
permafrost thaw, and use them to predict thaw type across Alaska.

## Glossary

Concepts only — one-line meanings. Symbols, units, and values live downstream.

- **Abrupt thaw** — rapid, often self-reinforcing permafrost thaw (class `0`, the majority ~93% of labeled points).
- **Gradual thaw** — slow, diffuse permafrost thaw (class `1`, the minority ~7%). Note: the training scripts treat class `1` as the metric "positive" (`predict_proba[:, 1]`), so "positive class" refers to Gradual, not Abrupt.
- **Thaw mode** — *which* of the two thaw pathways is occurring (abrupt vs. gradual); the classification target. Distinct from thaw *stage* (how far thaw has progressed) and thaw *occurrence* (whether a given hazard forms at all).
- **Sample prevalence vs. landscape prevalence** — the ~93%/7% abrupt/gradual split in the Thaw Database is *sample* prevalence, an artifact of lake- and road-biased collection; it is not *landscape* prevalence (the true areal fraction of each thaw mode across Alaska's permafrost), which is unknown and unrecoverable from this biased sample. The gap is why probabilities calibrated to the sample prior are suspect and why no single decision threshold is defensible.
- **Thaw Database** — the labeled point dataset providing each site's thaw-type label.
- **Feature table** — per-point geospatial features extracted and cleaned into model-ready training data.
- **SHAP values** — Shapley-value attributions used to rank and interpret each feature's contribution.
- **Prediction datacube** — the gridded statewide feature stack the model scores to produce thaw maps.
- **Abrupt-thaw susceptibility** — the model's continuous **log-evidence (likelihood-ratio) index** for abrupt (vs. gradual) thaw at a location: how strongly local features favor abrupt over gradual, on an absolute, prior-free scale (`0` = neutral, `>0` favors abrupt). It is **not** a calibrated probability (the sample prior is a sampling artifact; the landscape prior is unrecoverable) and **not** a discrete class. Formed as `logit(P_model(abrupt|x)) − logit(π_sample)`. Contrasts with categorical thermokarst-landscape classes.
- **Thermokarst landscape** — categorical susceptibility class (after Olefeldt et al. 2016) spanning lake/wetland/hillslope thaw forms; the incumbent map product this project's continuous surface responds to.

## Repo rules

- **Class encoding is fixed: `0 = Abrupt` (majority ~93%), `1 = Gradual` (minority ~7%).**
  This has been miscoded before (legacy artifacts use the reverse) — verify the
  encoding whenever touching labels, `predict_proba` indexing, class names, or
  confusion-matrix ordering. Ground truth is `clean_feature_table.py`:
  `Class = np.where(ThawType == 'Abrupt', 0, 1)`.
