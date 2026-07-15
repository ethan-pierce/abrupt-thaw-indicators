"""Ground the block geometry + dead-zone buffer for spatial block CV.

Consumes models/spatial_cv.py (the production splitter) at the SAME geometry the
training pipeline uses -- Alaska Albers equal-area blocks (`albers_grid`), 5-fold,
seed 42 -- so any buffer read off here is literally the buffer train_xgboost.py
will impose. Three questions:
  (1) GEOMETRY: for the pipeline's albers_grid cell sizes, how do the 1,107
      Non-abrupt (minority) points spread across folds? A fold with ~no positives
      makes AUC-PR unstable -> informs the block-size sweep.
  (2) BUFFER (T43): sweep the dead-zone buffer at the OPERATIVE block size
      (albers_grid, 10 km -> interpolation/case A, the map-serving regime) with a
      matched-count random-removal CONTROL. Unlike point-buffering, block holdout
      keeps the training set intact, so the targeted curve can actually plateau,
      and the targeted-vs-control gap isolates leakage from mere data loss. The
      operative buffer is where the targeted AUC-PR plateaus with the control
      confirming a real leakage-driven drop (positive gap). A flat curve with ~no
      gap is a valid finding: block holdout already removes the leakage, so
      BUFFER_KM = 0 is defensible (no nominal-scale floor -- T43).
  (3) A vs B: honest AUC-PR for small-block (interpolation) vs large-block
      (extrapolation) geometry at the buffer chosen in (2).

Positive class = 1 (Non-abrupt, minority); headline metric AUC-PR (chance = prevalence).
Estimator: a fixed regularized XGBoost config shared with leakage_decay.py (NOT the
operative-selected hyperparameters, which are stale pending the retrain, and would
be circular: the buffer feeds the CV the retrain runs). scale_pos_weight=balanced
matches leakage_decay for cross-probe continuity; it differs from the pipeline's
scale_pos_weight=1 (T10), but the buffer *range* is a property of the data's spatial
autocorrelation, not the estimator, so the read-off is unaffected.

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
OPERATIVE_CELL_KM = 10                 # pipeline OPERATIVE_CELL_KM (case A, serves the map)
SWEEP_CELL_KM = [10, 25, 50, 100, 200]  # pipeline SWEEP_CELL_KM (interpolation -> extrapolation)
N_SPLITS = 5                           # pipeline N_OUTER
RADII_KM = list(range(0, 16))          # 0..15 km, 1 km steps
GAP_EPS = 0.02                         # gap above this == leakage above noise


def factory(ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, min_child_weight=20, learning_rate=0.05,
        reg_lambda=50, gamma=1, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
        scale_pos_weight=spw, random_state=SEED)


def _score_folds(X, y, folds):
    """Pooled out-of-fold AUC-PR/ROC + %scored for a materialized fold list."""
    proba, scored = scv.pooled_oof_predict(factory, X, y, folds)
    yv = np.asarray(y)[scored]
    p = proba[scored]
    if len(np.unique(yv)) < 2:
        return np.nan, np.nan, scored.mean()
    return average_precision_score(yv, p), roc_auc_score(yv, p), scored.mean()


def _targeted_and_control_folds(lat, lon, base_folds, buffer_km, ctl_rng):
    """Build targeted (buffered) + matched-count random-control folds for one buffer.

    `base_folds` are the buffer=0 folds from the production splitter, so the block
    partition/test sets are IDENTICAL to what buffered_block_folds(buffer_km) yields
    -- applying `within_radius_of_set` here reproduces the production buffer exactly
    while letting us count removals to match the control.

      targeted: candidate-train MINUS points within buffer_km of any test point.
      control : candidate-train MINUS the SAME COUNT removed at random (spatially
                untargeted). The targeted-vs-control gap isolates leakage.
    """
    targeted, control, n_removed = [], [], 0
    for cand_train, test_idx in base_folds:
        if buffer_km > 0 and len(test_idx):
            in_zone = scv.within_radius_of_set(lat[cand_train], lon[cand_train],
                                               lat[test_idx], lon[test_idx], buffer_km)
        else:
            in_zone = np.zeros(len(cand_train), dtype=bool)
        n_rm = int(in_zone.sum())
        n_removed += n_rm
        targeted.append((cand_train[~in_zone], test_idx))
        if n_rm > 0:
            drop = ctl_rng.choice(len(cand_train), size=n_rm, replace=False)
            keep = np.ones(len(cand_train), dtype=bool)
            keep[drop] = False
            control.append((cand_train[keep], test_idx))
        else:
            control.append((cand_train, test_idx))
    return targeted, control, n_removed


def geometry(lat, lon, yv):
    print("=" * 74)
    print("(1) GEOMETRY -- albers_grid blocks, and Non-abrupt points per 5-fold split")
    print("=" * 74)
    print(f"{'cell_km':>8}{'#blocks':>9}{'pos/fold min':>14}{'median':>9}{'max':>7}{'folds w/0 pos':>15}")
    for cell_km in SWEEP_CELL_KM:
        blocks = scv.assign_blocks(lat, lon, method='albers_grid', cell_km=cell_km)
        uniq = np.unique(blocks)
        rng = np.random.default_rng(SEED)
        rng.shuffle(uniq)
        fold_of = {b: i % N_SPLITS for i, b in enumerate(uniq)}
        pos_per_fold = np.zeros(N_SPLITS, int)
        for b, f in fold_of.items():
            pos_per_fold[f] += int(((blocks == b) & (yv == 1)).sum())
        print(f"{cell_km:>8}{len(uniq):>9}{pos_per_fold.min():>14}"
              f"{int(np.median(pos_per_fold)):>9}{pos_per_fold.max():>7}"
              f"{int((pos_per_fold == 0).sum()):>15}")


def buffer_sweep(X, yv, lat, lon, prev):
    print("\n" + "=" * 74)
    print(f"(2) DEAD-ZONE BUFFER sweep  (albers_grid {OPERATIVE_CELL_KM} km, pooled OOF)")
    print("=" * 74)
    blocks = scv.assign_blocks(lat, lon, method='albers_grid', cell_km=OPERATIVE_CELL_KM)
    base = list(scv.buffered_block_folds(lat, lon, blocks, n_splits=N_SPLITS,
                                         buffer_km=0.0, seed=SEED))
    total_cand = sum(len(ct) for ct, _ in base)  # train-fold slots (buffer=0)

    print(f"{'buffer_km':>10}{'tgt_AUCPR':>11}{'ctl_AUCPR':>11}{'gap':>8}"
          f"{'removed%':>10}{'tgt_ROC':>9}")
    print("-" * 74)
    results = []
    for bkm in RADII_KM:
        ctl_rng = np.random.default_rng(1000 + bkm)
        targeted, control, n_removed = _targeted_and_control_folds(lat, lon, base,
                                                                   float(bkm), ctl_rng)
        tgt_ap, tgt_roc, _ = _score_folds(X, yv, targeted)
        if n_removed == 0:
            ctl_ap = tgt_ap
        else:
            ctl_ap, _, _ = _score_folds(X, yv, control)
        gap = ctl_ap - tgt_ap if np.isfinite(ctl_ap) and np.isfinite(tgt_ap) else np.nan
        removed_pct = 100 * n_removed / total_cand
        results.append((bkm, tgt_ap, ctl_ap, gap, tgt_roc, removed_pct))
        print(f"{bkm:>10}{tgt_ap:>11.4f}{ctl_ap:>11.4f}{gap:>8.4f}"
              f"{removed_pct:>9.1f}%{tgt_roc:>9.4f}")

    # ---- read-off: plateau of the targeted curve, confirmed by a real leakage gap.
    tgt = np.array([r[1] for r in results], dtype=float)
    roc = np.array([r[4] for r in results], dtype=float)
    gaps = np.array([r[3] for r in results], dtype=float)
    # Informative regime only: model still discriminates and stays above chance.
    informative = (roc > 0.55) & (tgt > prev + 0.02) & np.isfinite(tgt)
    plateau_r, pl_ap = None, None
    for i in range(len(tgt) - 3):
        w = tgt[i:i + 4]
        if (np.all(np.isfinite(w)) and informative[i:i + 4].all()
                and (w.max() - w.min()) < 0.01):
            plateau_r = results[i][0]
            pl_ap = tgt[i]
            break

    r0_ap = results[0][1]
    # The gap is the PRIMARY signal (Q3): leakage only counts where targeted removal
    # costs more AUC-PR than matched random removal, within the informative regime.
    inf_gaps = gaps[informative] if informative.any() else gaps
    max_gap = np.nanmax(inf_gaps) if np.isfinite(inf_gaps).any() else np.nan
    print("\n" + "=" * 74)
    print(f"block-CV (buffer=0) AUC-PR = {r0_ap:.4f}   max targeted-vs-control gap = {max_gap:.4f}")
    if not np.isfinite(max_gap) or max_gap < GAP_EPS:
        chosen = 0
        print(f"=> no targeted-vs-control gap exceeds {GAP_EPS} at any buffer: block "
              f"holdout at {OPERATIVE_CELL_KM} km already removes near-seam leakage.")
        print(f"   BUFFER_KM = 0 is defensible (no nominal-scale floor, T43).")
    elif plateau_r is not None:
        chosen = plateau_r
        print(f"targeted curve plateaus at ~{plateau_r} km (AUC-PR {pl_ap:.4f}, "
              f"drop {r0_ap - pl_ap:+.4f}); targeted < control confirms real leakage.")
        print(f"=> recommended operative buffer ~= {plateau_r} km.")
    else:
        chosen = 0
        print("leakage gap present but the targeted curve has not plateaued by 15 km -- "
              "extend the sweep before setting BUFFER_KM (leaving BUFFER_KM=0 for now).")

    try:
        _plot(results, prev, r0_ap, chosen, pl_ap, max_gap)
    except Exception as e:
        print(f"(plot skipped: {e})")
    return chosen


def a_vs_b(X, yv, lat, lon, buffer_km):
    print("\n" + "=" * 74)
    print(f"(3) INTERPOLATION (A, small blocks) vs EXTRAPOLATION (B, large blocks) "
          f"@ buffer {buffer_km} km")
    print("=" * 74)
    for label, cell_km in [("A  albers_grid  10 km", 10), ("B  albers_grid 100 km", 100),
                           ("B  albers_grid 200 km", 200)]:
        blocks = scv.assign_blocks(lat, lon, method='albers_grid', cell_km=cell_km)
        folds = list(scv.buffered_block_folds(lat, lon, blocks, n_splits=N_SPLITS,
                                              buffer_km=float(buffer_km), seed=SEED))
        ap, roc, frac = _score_folds(X, yv, folds)
        print(f"{label:<24} AUC-PR={ap:.4f}  AUC-ROC={roc:.4f}  scored={100*frac:.1f}%")


def _plot(results, prev, r0_ap, chosen, pl_ap, max_gap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = np.array([x[0] for x in results], dtype=float)
    tgt = np.array([x[1] for x in results], dtype=float)
    ctl = np.array([x[2] for x in results], dtype=float)
    removed = np.array([x[5] for x in results], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Shade the targeted<->control gap: this band is the leakage being removed
    # (targeted drops below control only because it strips spatially-leaky rows).
    finite = np.isfinite(tgt) & np.isfinite(ctl)
    ax.fill_between(r, tgt, ctl, where=finite, color='tab:orange', alpha=0.20,
                    label='leakage (targeted vs. control gap)')
    ax.plot(r, tgt, 'o-', lw=2, ms=4, color='tab:blue',
            label='targeted removal (buffer) = honest')
    ax.plot(r, ctl, 's--', lw=2, ms=4, color='gray',
            label='random removal (matched n) = control')
    ax.axhline(prev, color='k', ls=':', lw=1, label=f'AUC-PR chance floor ({prev:.3f})')

    if r0_ap is not None and np.isfinite(r0_ap):
        ax.annotate(f'buffer=0: {r0_ap:.3f}', xy=(0, r0_ap), xytext=(2, r0_ap + 0.06),
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1))
    # Reflect the ACTUAL verdict, not the raw plateau: only mark a recommended
    # buffer when a real leakage gap selected one; otherwise state the null result.
    if chosen and chosen > 0:
        ax.axvline(chosen, color='tab:green', ls='-', lw=1.4, alpha=0.8)
        if pl_ap is not None and np.isfinite(pl_ap):
            ax.annotate(f'recommended buffer ~{chosen} km: {pl_ap:.3f}',
                        xy=(chosen, pl_ap), xytext=(chosen + 1.5, pl_ap - 0.12),
                        fontsize=9, color='tab:green',
                        arrowprops=dict(arrowstyle='->', color='tab:green', lw=1))
    else:
        ax.text(0.5, prev + 0.18,
                f'targeted ≈ control at every buffer\n'
                f'max gap {max_gap:.3f} < 0.02 → no near-seam leakage\n'
                f'beyond block holdout → BUFFER_KM = 0',
                fontsize=9, color='0.25',
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))

    ax.set_title(f'Block-CV buffer decay: OOF AUC-PR vs dead-zone buffer '
                 f'(albers_grid {OPERATIVE_CELL_KM} km)')
    ax.set_xlabel('dead-zone buffer around held-out block (km)')
    ax.set_ylabel('AUC-PR (positive = Non-abrupt)')
    ax.legend(loc='center right', fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(r, removed, color='tab:red', alpha=0.35, lw=1)
    ax2.set_ylabel('% of training-fold points removed', color='tab:red')
    ax2.set_ylim(0, max(5, float(np.nanmax(removed)) * 1.2))
    fig.tight_layout()
    out = 'diagnostics/block_buffer_decay.png'
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    X, y, lat, lon = load(verify=True)
    yv = y.to_numpy()
    prev = yv.mean()
    n_pos = int((yv == 1).sum())
    print(f"n={len(y)}  Non-abrupt(minority)={n_pos}  prevalence={prev:.4f}  "
          f"AUC-PR floor={prev:.4f}\n")

    geometry(lat, lon, yv)
    chosen = buffer_sweep(X, yv, lat, lon, prev)
    a_vs_b(X, yv, lat, lon, chosen)

    print("\nCross-check vs the random-split probe (leakage_decay.py): random-split "
          "AUC-PR=0.904 (leaky), leakage-specific gap contiguous through ~2 km.")


if __name__ == '__main__':
    main()
