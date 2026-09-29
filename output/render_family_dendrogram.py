"""Re-render the SHAP feature-family dendrogram (appendix figure) without recomputing SHAP.

The clustering depends only on feature-space Spearman correlation, so the tree, cut, and
memberships are reproducible from features_clean.csv via `build_families` from
models/shap_groups.py; `models/shap_groups.py` produces the same figure.

Run: poetry run python output/render_family_dendrogram.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'models'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from settings import DATA, OUTPUT  # noqa: E402
from shap_values import load_inputs  # noqa: E402
from shap_groups import build_families, plot_dendrogram, MANUSCRIPT_LABELS  # noqa: E402
import shap_family_display as fd  # noqa: E402


def scored_members(fam_json):
    """Every feature that entered the grouping, from the recorded family memberships."""
    members = set()
    for fam in json.loads(Path(fam_json).read_text())['families']:
        members.update(fam['members'])
    return members


def main():
    members = scored_members(OUTPUT / 'shap_families.json')
    X, _y, _lat, _lon = load_inputs(DATA / 'features_clean.csv')

    missing = members - set(X.columns)
    if missing:
        raise SystemExit(f"features_clean.csv is missing scored columns: {sorted(missing)}")

    # Native CSV column order fixes the leaf layout.
    X_scored = X[[c for c in X.columns if c in members]]
    families, meta = build_families(X_scored)

    # Continuous families only: mapped name for multi-member clusters, column name for singletons.
    labels_by_key = {}
    for key, mem in families.items():
        if not key.startswith('cont_'):
            continue
        if len(mem) == 1:
            label = mem[0]
        else:
            label = MANUSCRIPT_LABELS.get(frozenset(mem))
            if label is None:
                raise SystemExit(
                    f"cluster {sorted(mem)} has no MANUSCRIPT_LABELS entry -- the cut changed; "
                    "reconcile models/shap_groups.py before re-rendering.")
        labels_by_key[key] = fd.NAMES[label]

    plot_dendrogram(meta, families, labels_by_key, OUTPUT)
    n_cont_fam = sum(1 for k in families if k.startswith('cont_'))
    print(f"Wrote {OUTPUT / 'shap_family_dendrogram.png'}  "
          f"(cut {meta['threshold']:.3f}; {len(meta['continuous'])} continuous leaves, "
          f"{n_cont_fam} continuous families)")


if __name__ == '__main__':
    main()
