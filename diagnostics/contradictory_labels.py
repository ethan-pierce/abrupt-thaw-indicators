"""T44 -- Contradictory-label ceiling: an irreducible bound on separation.

Some sites are *feature-identical yet label-disagreeing*: two training rows carry the
exact same model-feature vector but opposite thaw labels (one Abrupt=0, one Non-abrupt=1).
No classifier -- XGBoost, logistic, or a hypothetical oracle -- can separate them, because
the model only ever sees the features. They are irreducible noise, and they put a hard
CEILING on any separation metric (AUC-PR, accuracy) this project can report.

Why they survive into the training table: `clean_feature_table.py` deduplicates on the
feature columns *and* Class jointly (see diagnostics/_data.py). So identical rows with the
SAME label collapse to one, but identical rows with OPPOSITE labels both survive -- exactly
the contradictions. In the deduped table each contradiction therefore appears as a clean
1:1 pair (one 0, one 1); since Class is binary and (features, Class) is unique after dedup,
a feature-identical group has at most two members. NOTE: dedup collapses same-(feature,label)
MULTIPLICITY, so a raw 5-Abrupt / 1-Non-abrupt cluster reads here as a 1:1 tie. That is the
operative reality for the fitted model -- it too trains and is scored on this deduped table
(train_xgboost.py reads features_clean.csv directly) -- so the deduped 1:1 view is the
ceiling that actually bounds the REPORTED AUC-PR.

Two ceilings are reported:
  - ACCURACY ceiling: a deterministic model scores both members of a 0/1 pair identically,
    so it misclassifies exactly one per pair -> max accuracy = 1 - n_pairs / n_rows.
  - AUC-PR ceiling (positive = Non-abrupt): an ORACLE that perfectly ranks every uniquely-
    identified point (score = its true label) but is forced to 0.5 on every contradictory
    pair. Its average precision is the highest AUC-PR attainable if feature-identity were
    the ONLY source of confusion -- an optimistic upper bound (real features separate the
    rest imperfectly), i.e. a true ceiling. It contexts the reported GBM AUC-PR and the
    expected narrow GBM-vs-logistic margin: both models hit the same wall here.

Run: poetry run python diagnostics/contradictory_labels.py
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import load


def _row_keys(X):
    """Hashable per-row key over all feature columns, NaN normalized to a sentinel so
    NaN == NaN (matching pandas' dedup semantics that produced features_clean.csv)."""
    keys = []
    for row in X.itertuples(index=False, name=None):
        keys.append(tuple(None if (v != v) else v for v in row))  # v != v is True iff NaN
    return keys


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    X, y, lat, lon = load(verify=True)
    y = np.asarray(y)
    n_rows = len(y)
    n_pos = int((y == 1).sum())  # Non-abrupt (minority, the AUC-PR positive)

    groups = defaultdict(list)
    for i, k in enumerate(_row_keys(X)):
        groups[k].append(i)
    contradictions = [idx for idx in groups.values() if len(idx) > 1]

    # Structural check: each contradictory group must be exactly one 0 and one 1.
    for idx in contradictions:
        classes = sorted(y[idx].tolist())
        assert classes == [0, 1], (
            f"unexpected feature-identical group {classes} (size {len(idx)}); dedup on "
            "(features, Class) should make every such group a single 0/1 pair")

    n_pairs = len(contradictions)
    paired_idx = np.array([i for idx in contradictions for i in idx], dtype=int)
    n_involved = len(paired_idx)

    print("=" * 74)
    print("T44 -- Contradictory-label ceiling on separation (features_clean.csv)")
    print("=" * 74)
    print(f"Rows (deduped model table)      : {n_rows}")
    print(f"  Abrupt (0)                    : {int((y == 0).sum())}")
    print(f"  Non-abrupt (1, AUC-PR positive): {n_pos}")
    print(f"Distinct feature vectors        : {len(groups)}")
    print()
    print(f"Feature-identical / label-disagreeing pairs : {n_pairs}")
    print(f"Rows involved (2 per pair)                  : {n_involved} "
          f"({100 * n_involved / n_rows:.2f}% of rows)")
    print(f"Non-abrupt rows in a contradiction          : {n_pairs} "
          f"({100 * n_pairs / n_pos:.2f}% of the {n_pos} minority points)")

    if n_pairs == 0:
        print("\nNo feature-identical contradictions -> feature-identity imposes no ceiling.")
        print("=" * 74)
        return

    # Geographic context: contradictions arise from co-located points sharing a source
    # pixel, so pair members should sit ~atop each other.
    seps = np.array([_haversine_km(lat[i], lon[i], lat[j], lon[j])
                     for i, j in contradictions])
    print()
    print("Within-pair separation (mechanism = shared source pixel):")
    print(f"  median {np.median(seps):.3f} km | max {seps.max():.3f} km | "
          f"within 1 km: {100 * (seps <= 1.0).mean():.1f}%")

    # Ceiling 1: accuracy. One wrong per 0/1 pair, unavoidable.
    acc_ceiling = 1.0 - n_pairs / n_rows
    # Ceiling 2: AUC-PR of the oracle (true label everywhere except 0.5 on paired points).
    oracle = y.astype(float).copy()
    oracle[paired_idx] = 0.5
    aucpr_ceiling = average_precision_score(y, oracle)
    baseline_ap = n_pos / n_rows  # chance AUC-PR = prevalence

    print()
    print("Irreducible ceilings (feature-identity is the only confusion):")
    print(f"  Accuracy ceiling : {acc_ceiling:.6f}  (best case; 1 error per pair)")
    print(f"  AUC-PR ceiling   : {aucpr_ceiling:.6f}  (oracle; chance = {baseline_ap:.4f})")
    print()
    # Interpretation adapts to the magnitude: a near-1.0 ceiling is a NEGATIVE result --
    # hard contradictions are too few to explain any AUC-PR gap or a narrow model margin.
    if n_pairs / n_pos < 0.01:
        print("Reading: EXACT feature-identity contradictions are negligible here, so they")
        print(f"do NOT bound separation (ceiling {aucpr_ceiling:.4f} ~ 1.0). The gap the GBM")
        print("leaves to a perfect AUC-PR -- and the narrow GBM-vs-logistic margin -- is NOT")
        print("explained by hard label noise; it must come from SOFT overlap (points close")
        print("but not identical in feature space with opposing labels), which exact-match")
        print("does not capture. A near-neighbour / feature-overlap ceiling would quantify")
        print("that soft limit (needs a feature-distance metric choice -- separate probe).")
    else:
        print("Reading: any reported AUC-PR at/near the ceiling means the model has already")
        print("extracted essentially all separable signal; the gap to a perfect score (and")
        print("the narrowness of the GBM-vs-logistic margin) is bounded by irreducible label")
        print("noise, not model capacity.")
    print("=" * 74)


if __name__ == '__main__':
    main()
