"""Demonstrate the extrapolation-granularity range (case B).

Leave-one-region-out buffered block CV at increasing region SIZE. Coarser regions
force the model to predict farther from any training point, so AUC-PR degrades
along a physically meaningful axis: the median great-circle distance from a
held-out point to its nearest training point (the actual extrapolation reach).

Anchored by two computed reference points on the same data/model:
  * random split (leaky, co-located train/test)  -- upper bound
  * interpolation (A: grid 0.5deg + 3 km buffer)  -- the map metric
...down toward the AUC-PR chance floor (= prevalence).

Consumes models/spatial_cv.py. Positive class = 1 (Non-abrupt). Pooled OOF AUC-PR.
Run: poetry run python diagnostics/extrapolation_range.py
"""
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'models'))
import spatial_cv as scv
from _data import load

SEED = 42
BUFFER_KM = 3
GRANULARITIES = [3, 5, 7, 10, 15, 20, 30, 50]   # kmeans regions: few=coarse/far, many=fine/near


def factory(ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, min_child_weight=20, learning_rate=0.05,
        reg_lambda=50, gamma=1, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
        scale_pos_weight=spw, random_state=SEED)


def score_folds(X, y, folds):
    proba, scored = scv.pooled_oof_predict(factory, X, y, folds)
    yv = y[scored]
    p = proba[scored]
    return average_precision_score(yv, p), roc_auc_score(yv, p), scored.mean()


def extrap_distances(lat, lon, folds):
    """Median / IQR great-circle distance (km) from held-out points to nearest train point."""
    d = []
    for tr, te in folds:
        if len(tr) == 0 or len(te) == 0:
            continue
        tree = BallTree(np.radians(np.column_stack([lat[tr], lon[tr]])), metric='haversine')
        dist, _ = tree.query(np.radians(np.column_stack([lat[te], lon[te]])), k=1)
        d.append(dist.ravel() * scv.EARTH_KM)
    d = np.concatenate(d)
    return np.median(d), np.percentile(d, 25), np.percentile(d, 75)


def main():
    X, y, lat, lon = load(verify=True)
    yv = y.to_numpy()
    prev = yv.mean()
    print(f"n={len(y)}  Non-abrupt={int((yv==1).sum())}  prevalence={prev:.4f}  "
          f"buffer={BUFFER_KM}km  (AUC-PR floor={prev:.4f})\n")

    # --- reference: random split (leaky) ---
    rng = np.random.default_rng(SEED)
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.3,
                                random_state=int(rng.integers(0, 100)),
                                shuffle=True, stratify=y)
    m = factory(yv[itr]).fit(X.iloc[itr], yv[itr])
    rand_ap = average_precision_score(yv[ite], m.predict_proba(X.iloc[ite])[:, 1])

    # --- reference: interpolation (A) ---
    blocks_A = scv.assign_blocks(lat, lon, method='grid', cell_deg=0.5)
    folds_A = list(scv.buffered_block_folds(lat, lon, blocks_A, n_splits=5, buffer_km=BUFFER_KM, seed=SEED))
    A_ap, A_roc, _ = score_folds(X, yv, folds_A)
    A_dist = extrap_distances(lat, lon, folds_A)[0]

    # --- the extrapolation sweep ---
    print(f"{'regions':>8}{'med_dist_km':>13}{'IQR_km':>18}{'AUC-PR':>9}{'AUC-ROC':>9}")
    print("-" * 60)
    rows = []
    for g in GRANULARITIES:
        blocks = scv.assign_blocks(lat, lon, method='kmeans', n_clusters=g, seed=SEED)
        folds = list(scv.buffered_block_folds(lat, lon, blocks, n_splits=g, buffer_km=BUFFER_KM, seed=SEED))
        ap, roc, frac = score_folds(X, yv, folds)
        med, q25, q75 = extrap_distances(lat, lon, folds)
        rows.append((g, med, q25, q75, ap, roc))
        print(f"{g:>8}{med:>13.1f}{('['+format(q25,'.0f')+'-'+format(q75,'.0f')+']'):>18}{ap:>9.4f}{roc:>9.4f}")

    print("\nreference points (same data/model):")
    print(f"  random split (leaky)                 AUC-PR={rand_ap:.4f}")
    print(f"  interpolation A (grid0.5+{BUFFER_KM}km, ~{A_dist:.1f}km) AUC-PR={A_ap:.4f}  AUC-ROC={A_roc:.4f}")
    print(f"  chance floor                          AUC-PR={prev:.4f}")

    try:
        _plot(rows, rand_ap, (A_dist, A_ap), prev)
    except Exception as e:
        print(f"(plot skipped: {e})")


def _plot(rows, rand_ap, A_point, prev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    med = [r[1] for r in rows]
    ap = [r[4] for r in rows]
    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(med, ap, 'o-', lw=2, ms=6, color='tab:blue', label='extrapolation (leave-region-out)')
    for x, yv_, g in zip(med, ap, labels):
        ax.annotate(f'{g}', (x, yv_), textcoords='offset points', xytext=(0, 7), fontsize=8, ha='center')
    ax.plot(A_point[0], A_point[1], 'D', ms=9, color='tab:green', label=f'interpolation A ({A_point[1]:.2f})')
    ax.axhline(rand_ap, ls='--', color='tab:red', lw=1.2, label=f'random split, leaky ({rand_ap:.2f})')
    ax.axhline(prev, ls=':', color='k', lw=1, label=f'chance floor ({prev:.3f})')
    ax.set_xlabel('median extrapolation distance to nearest training point (km)')
    ax.set_ylabel('AUC-PR (positive = Non-abrupt)')
    ax.set_title('Extrapolation-granularity range: AUC-PR vs how far the model must reach\n'
                 '(labels = number of held-out regions)')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = 'diagnostics/extrapolation_range.png'
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == '__main__':
    main()
