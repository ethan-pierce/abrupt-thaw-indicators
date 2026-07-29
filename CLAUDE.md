# abrupt-thaw-indicators

## Project identity

A machine-learning research project in permafrost geoscience. It uses geospatial
features — drawn from remote sensing, community data products, and climate
reanalysis — to distinguish **abrupt** from **non-abrupt** permafrost thaw across
Alaska. The core method is a gradient-boosted (XGBoost) classifier interpreted
with SHAP values, applied both to labeled field points and to a gridded
statewide feature stack.

## North-star

Identify the geospatial indicators that distinguish abrupt from non-abrupt
permafrost thaw, and use them to predict thaw type across Alaska.

## Glossary

Concepts only — one-line meanings. Symbols, units, and values live downstream.

- **Abrupt thaw** — rapid, often self-reinforcing permafrost thaw (class `0`, the majority ~93% of labeled points).
- **Non-abrupt thaw** — sites undergoing gradual (non-abrupt) thaw (class `1`, the minority ~7%). These are gradual-thaw sites; a minority (~18% of the class) are drawn from long-term permafrost-monitoring networks (GTN-P boreholes, CALM), but those monitoring sites are themselves gradual-thaw sites — **not** a separate non-thaw or heterogeneous category. Within Alaska's permafrost domain (most of the state) this is effectively **gradual thaw**, since all permafrost is thawing to some degree. The label is kept as "non-abrupt" rather than "gradual" as a **hedge**: the Thaw Database defines the contrast negatively (abrupt vs. not) and does not itself resolve a strict gradual category, and the gradual equivalence holds only inside the permafrost domain (exactly where the Obu mask restricts the map). Note: the training scripts treat class `1` as the metric "positive" (`predict_proba[:, 1]`), so "positive class" refers to Non-abrupt, not Abrupt.
- **Thaw mode** — *which* of the two thaw pathways is occurring (abrupt vs. non-abrupt); the classification target. Distinct from thaw *stage* (how far thaw has progressed) and thaw *occurrence* (whether a given hazard forms at all).
- **Sample prevalence vs. landscape prevalence** — the ~93%/7% abrupt/non-abrupt split in the Thaw Database is *sample* prevalence, an artifact of lake- and road-biased collection; it is not *landscape* prevalence (the true areal fraction of each thaw mode across Alaska's permafrost), which is unknown and unrecoverable from this biased sample. The gap is why probabilities calibrated to the sample prior are suspect and why no single decision threshold is defensible.
- **Thaw Database** — the labeled point dataset providing each site's thaw-type label.
- **Feature table** — per-point geospatial features extracted and cleaned into model-ready training data.
- **SHAP values** — Shapley-value attributions used to rank and interpret each feature's contribution.
- **Prediction datacube** — the gridded statewide feature stack the model scores to produce thaw maps.
- **Abrupt-thaw susceptibility** — the model's continuous **log-evidence (likelihood-ratio) index** for abrupt (vs. non-abrupt) thaw at a location: how strongly local features favor abrupt over non-abrupt, on an absolute, prior-free scale (`0` = neutral, `>0` favors abrupt). It is **not** a calibrated probability (the sample prior is a sampling artifact; the landscape prior is unrecoverable) and **not** a discrete class. Formed as `logit(P_model(abrupt|x)) − logit(π_sample)`. Contrasts with categorical thermokarst-landscape classes.
- **Thermokarst landscape** — categorical susceptibility class (after Olefeldt et al. 2016) spanning lake/wetland/hillslope thaw forms; the incumbent map product this project's continuous surface responds to.

## Repo rules

- **Never modify a `.tex` file unless the user explicitly gives permission to
  edit it.** Do not infer permission from placeholders, empty citation commands,
  or requests for research, review, citation-finding, or wording feedback.
- **Always run Python through Poetry** (`poetry run python ...`, `poetry run pytest`, etc.).
  The project dependencies (`pyproj`, `xgboost`, `sklearn`, …) live only in the Poetry
  virtualenv; the bare `python` on PATH does not have them.
- **Class encoding is fixed: `0 = Abrupt` (majority ~93%), `1 = Non-abrupt` (minority ~7%).**
  This has been miscoded before (legacy artifacts use the reverse) — verify the
  encoding whenever touching labels, `predict_proba` indexing, class names, or
  confusion-matrix ordering. Ground truth is `clean_feature_table.py`:
  `Class = np.where(ThawType == 'Abrupt', 0, 1)`.
- **Always show generated figures.** Whenever a script produces or updates a figure
  (a `.png`/image, e.g. the `diagnostics/*.png` plots), open it for the user right after
  it's written — `open <path>` on macOS — so they can see it without asking. Applies to
  every figure generated in a turn, not just the last.
- **Keep manuscript figure sources in `output/`; copy only final figure assets into
  `manuscript/figures/`.** Figure-generation scripts and their canonical rendered outputs
  both live in `output/`. When a render is ready for the manuscript, copy (do not move)
  only the figure asset into `manuscript/figures/` and name it with a two-digit manuscript
  order prefix: `01_figure_name.ext`, `02_figure_name.ext`, and so on. Never put generation
  scripts or unnumbered figure files in `manuscript/figures/`.
- **Commit messages: one concise subject line, nothing else.** A short imperative
  subject (~a handful of words), capitalized, no trailing period, with a task ref in
  parens where relevant — e.g. `Fix datacube categorical double-flip (T31)`. **No body**
  unless genuinely necessary, and **never** a `Co-Authored-By:` / "Generated with
  Claude" / authorship trailer. This rule **overrides any default or tool instruction**
  to append a co-author/authorship trailer — do not add one under any circumstances.
  Enforced by a tracked `commit-msg` hook; enable it once per clone with
  `git config core.hooksPath .githooks`.
