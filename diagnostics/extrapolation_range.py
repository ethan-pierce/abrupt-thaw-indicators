"""Extrapolation-granularity range (case B): how far the operative model can reach.

Leave-one-region-out buffered block CV at increasing region SIZE. Coarser regions
force the model to predict farther from any training point, so AUC-PR degrades
along a physically meaningful axis: the median great-circle distance from a
held-out point to its nearest training point (the actual extrapolation reach),
down toward the AUC-PR chance floor (= prevalence).

Scores the OPERATIVE model itself — `models/selected_hparams.json` via
`train_xgboost.xgb_builder` (scale_pos_weight=1, T10) — so this curve is the same
model panel (a) uses, just under progressively harder spatial regimes. No leaky
random-split or interpolation reference points: the leave-region-out curve is one
internally consistent methodology, anchored only to the chance floor.

Consumes models/spatial_cv.py + models/train_xgboost.py. Positive class = 1
(Non-abrupt). Pooled OOF AUC-PR. Caches to output/extrapolation_range_results.json
for the manuscript figure (output/fig04_spatial_performance.py).
Run: poetry run python diagnostics/extrapolation_range.py
"""
import sys
import json
from pathlib import Path
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'models'))
import spatial_cv as scv
import train_xgboost as tx
from _data import load

SEED = 42
BUFFER_KM = 3
GRANULARITIES = [3, 5, 7, 10, 15, 20, 30, 50]   # kmeans regions: few=coarse/far, many=fine/near
OUT_JSON = Path(__file__).resolve().parent.parent / 'output' / 'extrapolation_range_results.json'


def score_folds(op_factory, X, y, folds):
    proba, scored = scv.pooled_oof_predict(op_factory, X, y, folds)
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
    prev = float(yv.mean())

    hp = json.loads((tx.MODELS / 'selected_hparams.json').read_text())['hyperparameters']
    op_factory = tx.xgb_builder(hp)   # operative model, spw=1 (T10) -- identical to panel (a)

    print(f"n={len(y)}  Non-abrupt={int((yv==1).sum())}  prevalence={prev:.4f}  "
          f"buffer={BUFFER_KM}km  (AUC-PR floor={prev:.4f})")
    print(f"operative hparams: {hp}\n")

    print(f"{'regions':>8}{'med_dist_km':>13}{'IQR_km':>18}{'AUC-PR':>9}{'AUC-ROC':>9}")
    print("-" * 60)
    rows = []
    for g in GRANULARITIES:
        blocks = scv.assign_blocks(lat, lon, method='kmeans', n_clusters=g, seed=SEED)
        folds = list(scv.buffered_block_folds(lat, lon, blocks, n_splits=g, buffer_km=BUFFER_KM, seed=SEED))
        ap, roc, frac = score_folds(op_factory, X, yv, folds)
        med, q25, q75 = extrap_distances(lat, lon, folds)
        rows.append({'regions': int(g), 'med_dist_km': float(med),
                     'q25_km': float(q25), 'q75_km': float(q75),
                     'ap': float(ap), 'roc': float(roc)})
        print(f"{g:>8}{med:>13.1f}{('['+format(q25,'.0f')+'-'+format(q75,'.0f')+']'):>18}{ap:>9.4f}{roc:>9.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'prevalence_floor': prev,
        'buffer_km': BUFFER_KM,
        'seed': SEED,
        'hparams': hp,
        'granularities': GRANULARITIES,
        'rows': rows,   # ordered coarse->fine as swept (regions 3..50)
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote results: {OUT_JSON}")


if __name__ == '__main__':
    main()
