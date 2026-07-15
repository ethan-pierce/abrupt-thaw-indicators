"""Feature-provenance trace: does a single feature proxy the target?

A depth-1 stump reaches AUC~0.85 (baseline_and_shuffle.py), so one feature nearly
separates abrupt from non-abrupt. This scans every feature's univariate separating
power and flags any that behaves like a target proxy -- especially a water/lake
indicator, since the DB is lake-dominated (abrupt ~= thermokarst lake), which
would encode the SAMPLING DESIGN rather than a thaw mechanism (SCOPE.md, README #13).

Run: poetry run python diagnostics/feature_provenance.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from _data import load

X, y, lat, lon = load(verify=True)
yv = y.to_numpy()
prev = y.mean()
print(f"n={len(y)}  positive(Non-abrupt) prevalence={prev:.4f}  (AUC-PR floor={prev:.4f})\n")

rows = []
for col in X.columns:
    v = X[col].to_numpy(dtype=float)
    m = np.isfinite(v)
    if m.sum() < 100 or len(np.unique(v[m])) < 2:
        continue
    # orient score so higher = positive class; report max(auc, 1-auc)
    auc = roc_auc_score(yv[m], v[m])
    signed = max(auc, 1 - auc)
    ap = average_precision_score(yv[m], v[m] if auc >= 0.5 else -v[m])
    rows.append((col, signed, ap, m.mean()))

rows.sort(key=lambda r: r[2], reverse=True)
print(f"{'feature':<42} {'|AUC|':>7} {'AUC-PR':>8} {'coverage':>9}")
print("-" * 70)
for col, auc, ap, cov in rows[:20]:
    flag = "  <-- strong single-feature separator" if ap > 0.3 else ""
    print(f"{col:<42} {auc:>7.4f} {ap:>8.4f} {cov:>8.1%}{flag}")

print("\nTop separators that are land-cover / water / location proxies are suspect: "
      "\nthey may encode the lake-biased SAMPLING design, not a thaw mechanism.")
