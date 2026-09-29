# Repo rules

- **Run Python through Poetry** (`poetry run python ...`, `poetry run pytest`). Dependencies live only in the Poetry virtualenv.
- **Never modify a `.tex` file** unless the user explicitly gives permission to edit it. Placeholders, empty citations, or requests for research, review, or wording feedback are not permission.
- **Class encoding is fixed: `0 = Abrupt` (majority), `1 = Non-abrupt` (minority).** Verify it whenever touching labels, `predict_proba` indexing, class names, or confusion-matrix ordering. Ground truth is `data/clean_feature_table.py`: `Class = np.where(ThawType == 'Abrupt', 0, 1)`.
- **Show every generated figure.** After a script writes or updates an image, open it for the user (`open <path>` on macOS).
- **Figure sources and renders live in `output/`.** Copy (never move) a final figure asset into `manuscript/figures/` with a two-digit order prefix (`01_figure_name.pdf`), and only after the user explicitly approves that specific figure. Never put scripts or unnumbered files in `manuscript/figures/`.
- **Commit messages are one short imperative subject line**, capitalized, with no trailing period, no body, and no trailers of any kind (no `Co-Authored-By`, sign-off, or attribution).
