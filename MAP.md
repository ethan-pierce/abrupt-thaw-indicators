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
| `settings.py` | Path configuration (`ROOT`, `DATA`, `MODELS`, `OUTPUT`). |
| `pyproject.toml` | Poetry dependencies and project metadata. |
| `data/` | Data-processing and feature-extraction scripts, databases, and datasets. |
| `models/` | Model training, prediction, and SHAP-interpretation scripts plus serialized models. |
| `output/` | Generated figures, maps, and result artifacts. |
| `archive/` | Legacy code, older data, and experimental work. |
