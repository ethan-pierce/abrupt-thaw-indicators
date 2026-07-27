# MAP

Locations index. Every skill's Orient reads this. Rows are locations and their
roles only — no claims, no science. Downstream skills add rows as they register
their own artifacts.

| Location | Role |
| --- | --- |
| `CLAUDE.md` | Project backbone: identity, north-star, glossary, repo rules. |
| `MAP.md` | This index. |
| `SCOPE.md` | Manuscript scope: key background, objectives, brainstorm, open blockers. |
| `REFERENCES.md` | Literature record: sources actually opened, keyed for citation by other docs. |
| `README.md` | Human-facing overview, setup, usage, and the manuscript-prep to-do list. |
| `PIPELINE.md` | End-to-end provenance: data sources, script→artifact DAG in order, final outputs. |
| `archive/TASKS.md` | Closed ledger of the methods-cleanup + feature-rebuild tasks (all complete); the decoder for `TNN` references cited across the repo. |
| `settings.py` | Path configuration (`ROOT`, `DATA`, `MODELS`, `OUTPUT`). |
| `pyproject.toml` | Poetry dependencies and project metadata. |
| `data/` | Data-processing and feature-extraction scripts, databases, and datasets. |
| `models/` | Model training, prediction, and SHAP-interpretation scripts plus serialized models. |
| `models/spatial_cv.py` | Buffered spatial-block CV splitter (block size selects interpolation vs extrapolation); the honest-split protocol `train_xgboost.py` uses in place of a random split. |
| `models/aoa.py` | Area-of-Applicability reliability layer; runs after `predict.py`, emits `data/aoa.nc` + CV-calibrated threshold. |
| `output/` | Figure-generation scripts and rendered figure assets (plus the `figstyle.py`/`STYLE.md` style guide). |
| `diagnostics/` | `/verify-ml` suite: re-runnable baseline, shuffle-label, spatial-leakage, and feature-provenance probes, plus the stamped findings report. |
| `diagnostics/FINDINGS.md` | Stamped `/verify-ml` findings for the current pipeline: invariants that hold + the honest-generalization flags. |
| `manuscript/` | Earth's Future draft: `main.tex` (+ `sections/`, `figures/`), with `STRATEGY.md` (argument spine), `OUTLINE.md` (prose skeleton), and `figures/FIGURES.md` (figure spec). |
| `archive/` | Legacy code, older data, and experimental work. |
