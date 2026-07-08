"""Ground the block geometry + dead-zone buffer for spatial block CV.

Consumes models/spatial_cv.py (the production splitter). Three questions:
  (1) GEOMETRY: for candidate grid cell sizes and kmeans cluster counts, how do the
      1,103 Gradual (minority) points spread across folds? A fold with ~no positives
      makes AUC-PR unstable -> informs grid-vs-kmeans and block count.
  (2) BUFFER: sweep the dead-zone buffer under a fixed block geometry (pooled OOF
      AUC-PR). Unlike point-buffering, block holdout keeps the training set intact,
      so this curve can actually plateau -> reads off the honest buffer distance.
  (3) A vs B: honest AUC-PR for small-block (interpolation) vs large-block
      (extrapolation) geometry at the chosen buffer.

Run: poetry run python diagnostics/block_cv.py
"""
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'models'))
import spatial_cv as scv
from _data import load

SEED = 42


def factory(ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, min_child_weight=20, learning_rate=0.05,
        reg_lambda=50, gamma=1, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
        scale_pos_weight=spw, random_state=SEED)


def pooled(X, y, lat, lon, blocks, n_splits, buffer_km):
    folds = list(scv.buffered_block_folds(lat, lon, blocks, n_splits=n_splits,
                                          buffer_km=buffer_km, seed=SEED))
    proba, scored = scv.pooled_oof_predict(factory, X, y, folds)
    yv = np.asarray(y)[scored]
    p = proba[scored]
    ap = average_precision_score(yv, p) if len(np.unique(yv)) == 2 else np.nan
    roc = roc_auc_score(yv, p) if len(np.unique(yv)) == 2 else np.nan
    return ap, roc, scored.mean()


def main():
    X, y, lat, lon = load(verify=True)
    yv = y.to_numpy()
    prev = yv.mean()
    n_pos = int((yv == 1).sum())
    print(f"n={len(y)}  Gradual(minority)={n_pos}  prevalence={prev:.4f}  AUC-PR floor={prev:.4f}\n")

    # ---- (1) GEOMETRY: minority spread across folds ----
    print("=" * 74)
    print("(1) GEOMETRY -- blocks, and Gradual points per 5-fold split")
    print("=" * 74)
    print(f"{'geometry':<26}{'#blocks':>8}{'pos/fold min':>14}{'median':>9}{'max':>7}{'folds w/0 pos':>15}")
    geoms = [('grid 0.5deg', dict(method='grid', cell_deg=0.5)),
             ('grid 1.0deg', dict(method='grid', cell_deg=1.0)),
             ('grid 2.0deg', dict(method='grid', cell_deg=2.0)),
             ('kmeans 50', dict(method='kmeans', n_clusters=50)),
             ('kmeans 25', dict(method='kmeans', n_clusters=25)),
             ('kmeans 10', dict(method='kmeans', n_clusters=10)),
             ('kmeans 5', dict(method='kmeans', n_clusters=5))]
    for name, kw in geoms:
        blocks = scv.assign_blocks(lat, lon, seed=SEED, **kw)
        nb = len(np.unique(blocks))
        # distribute blocks to 5 folds exactly as the splitter does, count positives/fold
        uniq = np.unique(blocks)
        rng = np.random.default_rng(SEED)
        rng.shuffle(uniq)
        fold_of = {b: i % 5 for i, b in enumerate(uniq)}
        pos_per_fold = np.zeros(5, int)
        for b, f in fold_of.items():
            pos_per_fold[f] += int(((blocks == b) & (yv == 1)).sum())
        print(f"{name:<26}{nb:>8}{pos_per_fold.min():>14}{int(np.median(pos_per_fold)):>9}"
              f"{pos_per_fold.max():>7}{int((pos_per_fold == 0).sum()):>15}")

    # ---- (2) BUFFER sweep under a fixed small-block geometry (interpolation, A) ----
    print("\n" + "=" * 74)
    print("(2) DEAD-ZONE BUFFER sweep  (geometry: grid 0.5deg, pooled OOF AUC-PR)")
    print("=" * 74)
    blocks_A = scv.assign_blocks(lat, lon, method='grid', cell_deg=0.5)
    print(f"{'buffer_km':>10}{'AUC-PR':>10}{'AUC-ROC':>10}{'%scored':>10}")
    prev_ap = None
    for bkm in [0, 1, 2, 3, 5, 7, 10]:
        ap, roc, frac = pooled(X, yv, lat, lon, blocks_A, n_splits=5, buffer_km=bkm)
        flag = ""
        if prev_ap is not None and abs(ap - prev_ap) < 0.01:
            flag = "  <- plateau (dAUC-PR<0.01)"
        print(f"{bkm:>10}{ap:>10.4f}{roc:>10.4f}{100*frac:>9.1f}%{flag}")
        prev_ap = ap

    # ---- (3) A (interpolation) vs B (extrapolation) at a modest buffer ----
    print("\n" + "=" * 74)
    print("(3) INTERPOLATION (A, small blocks) vs EXTRAPOLATION (B, large blocks)")
    print("=" * 74)
    for label, blocks, ns, bkm in [
        ("A  grid 0.5deg  buffer 3km", scv.assign_blocks(lat, lon, method='grid', cell_deg=0.5), 5, 3),
        ("B  kmeans 10    buffer 3km", scv.assign_blocks(lat, lon, method='kmeans', n_clusters=10), 5, 3),
        ("B  kmeans 5     buffer 3km", scv.assign_blocks(lat, lon, method='kmeans', n_clusters=5), 5, 3),
    ]:
        ap, roc, frac = pooled(X, yv, lat, lon, blocks, ns, bkm)
        print(f"{label:<30} AUC-PR={ap:.4f}  AUC-ROC={roc:.4f}  scored={100*frac:.1f}%")

    print("\nCross-check vs verify-ml: random split AUC-PR=0.903 (leaky), "
          "point-buffer 1-3km 0.71-0.78, spatial GroupKFold 0.70-0.78.")


if __name__ == '__main__':
    main()
