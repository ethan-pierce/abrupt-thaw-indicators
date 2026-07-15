"""Repeated spatial block-CV: partition-to-partition spread of the AUC-PR-vs-scale curve.

The headline curve in `train_xgboost.py` is a SINGLE 5-fold block partition per scale, so
its across-fold spread mixes real spatial-generalization variance with the luck of one
block->fold draw. This repeats the draw `N_REPEATS` times per scale (varying ONLY the
block->fold shuffle seed; the albers_grid placement is fixed) and reports the distribution
of pooled-OOF AUC-PR -- the honest uncertainty a statewide extrapolating map inherits.

It also tracks the XGBoost-vs-logistic MARGIN per partition (both models scored on the
IDENTICAL folds each repeat -> a paired contrast), to test whether the narrow
operative-scale margin is a stable signal or within partition noise.

Fixed configs (NO per-fold re-selection) so the measured spread is PURE partition variance,
not hyperparameter-selection noise:
  - XGBoost: the operative selected hyperparameters (`models/selected_hparams.json`),
    scale_pos_weight=1 (T10) -- i.e. the operative model itself, held constant across scales.
  - Logistic: the T45 baseline pipeline (`logistic_builder`) at fixed C=1.0.
Single-level `N_OUTER`-fold block folds via the production splitter, buffer = BUFFER_KM (0,
T43). Only `CV_SEED` varies across repeats. Positive class = 1 (Non-abrupt); chance = prevalence.
Extension not done here: jittering the grid ORIGIN (block boundaries are fixed for
albers_grid) would add a second partition-variance source.

Run: poetry run python diagnostics/repeated_cv.py
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_cv as scv
import train_xgboost as tx
from _data import load

SCALES = [5, 10, 25, 50, 100, 200]   # matches SWEEP_CELL_KM (5 km bookend added, T-repeated-cv)
N_REPEATS = 20                        # block->fold reshuffles per scale
SEED0 = 42                            # repeat r uses CV seed SEED0 + r
LOGIT_C = 1.0                         # best fixed C at the operative scale (verified: C=0.01/0.1/1.0
                                      # -> 0.740/0.768/0.785 OOF AUC-PR @10 km), so the baseline is
                                      # its strongest fixed config, not a handicap. NOTE: the headline
                                      # pipeline SELECTS C per fold, which reaches ~0.81 @10 km (adaptive
                                      # > any single fixed C); this diagnostic fixes it to isolate
                                      # partition variance from selection variance for both models.
FIG = Path(__file__).resolve().parent / 'repeated_cv.png'


def pooled_ap(factory, X, y, folds):
    """Pooled out-of-fold AUC-PR for a fixed estimator factory over a fold list."""
    proba, scored = scv.pooled_oof_predict(factory, X, y, folds)
    yv = np.asarray(y)[scored]
    if len(np.unique(yv)) < 2:
        return np.nan
    return average_precision_score(yv, proba[scored])


def main():
    X, y, lat, lon = load(verify=True)
    yv = np.asarray(y)
    prevalence = float((yv == 1).mean())

    hp = json.loads((tx.MODELS / 'selected_hparams.json').read_text())['hyperparameters']
    xgb_factory = tx.xgb_builder(hp)                 # operative config, spw=1 (T10)
    logit_factory = tx.logistic_builder({'C': LOGIT_C})

    print("=" * 78)
    print(f"Repeated block-CV: {N_REPEATS} block->fold reshuffles/scale | "
          f"{tx.N_OUTER}-fold | buffer {tx.BUFFER_KM} km")
    print(f"XGBoost operative hparams: {hp}")
    print(f"Logistic baseline: T45 pipeline, C={LOGIT_C} | positive=Non-abrupt "
          f"(chance AUC-PR={prevalence:.4f})")
    print("=" * 78)
    print(f"{'cell_km':>8} | {'XGBoost AUC-PR mean±std [min,max]':^34} | "
          f"{'Logistic mean±std':^18} | {'margin mean±std':^16}")
    print("-" * 78)

    stats = {}
    for scale in SCALES:
        blocks = scv.assign_blocks(lat, lon, method=tx.BLOCK_METHOD, cell_km=scale)
        xgb_aps, logit_aps = [], []
        for r in range(N_REPEATS):
            folds = list(scv.buffered_block_folds(
                lat, lon, blocks, n_splits=tx.N_OUTER, buffer_km=tx.BUFFER_KM, seed=SEED0 + r))
            xgb_aps.append(pooled_ap(xgb_factory, X, y, folds))
            logit_aps.append(pooled_ap(logit_factory, X, y, folds))
        xgb_aps, logit_aps = np.array(xgb_aps), np.array(logit_aps)
        margin = xgb_aps - logit_aps  # paired: same folds each repeat
        stats[scale] = {
            'xgb_mean': np.nanmean(xgb_aps), 'xgb_std': np.nanstd(xgb_aps),
            'xgb_min': np.nanmin(xgb_aps), 'xgb_max': np.nanmax(xgb_aps),
            'logit_mean': np.nanmean(logit_aps), 'logit_std': np.nanstd(logit_aps),
            'margin_mean': np.nanmean(margin), 'margin_std': np.nanstd(margin),
        }
        s = stats[scale]
        print(f"{scale:>8} | {s['xgb_mean']:.3f} ± {s['xgb_std']:.3f} "
              f"[{s['xgb_min']:.3f}, {s['xgb_max']:.3f}]".ljust(37) +
              f"| {s['logit_mean']:.3f} ± {s['logit_std']:.3f}".ljust(21) +
              f"| {s['margin_mean']:+.3f} ± {s['margin_std']:.3f}")

    print("-" * 78)
    if tx.OPERATIVE_CELL_KM in stats:
        op = stats[tx.OPERATIVE_CELL_KM]
        margin_stable = abs(op['margin_mean']) > 2 * op['margin_std']
        print(f"Operative scale {tx.OPERATIVE_CELL_KM} km margin: {op['margin_mean']:+.3f} "
              f"± {op['margin_std']:.3f} -> XGBoost edge is "
              f"{'OUTSIDE' if margin_stable else 'WITHIN'} 2σ of partition noise "
              f"({'stable signal' if margin_stable else 'not distinguishable from noise'}).")

    # Figure: mean line + ±1σ band per model, prevalence floor.
    fig, ax = plt.subplots(figsize=(8, 5))
    cells = np.array(SCALES)
    for key, label, color in [('xgb', 'XGBoost (operative)', 'C0'),
                              ('logit', 'Logistic (T45 baseline)', 'C1')]:
        m = np.array([stats[c][f'{key}_mean'] for c in SCALES])
        sd = np.array([stats[c][f'{key}_std'] for c in SCALES])
        ax.plot(cells, m, marker='o', color=color, label=label)
        ax.fill_between(cells, m - sd, m + sd, color=color, alpha=0.2)
    ax.axhline(prevalence, color='grey', linestyle=':', label=f'prevalence floor ({prevalence:.3f})')
    ax.axvline(tx.OPERATIVE_CELL_KM, color='k', linestyle='--', alpha=0.4,
               label=f'operative {tx.OPERATIVE_CELL_KM} km')
    ax.set_xscale('log')
    ax.set_xticks(cells)
    ax.set_xticklabels([str(c) for c in SCALES])
    ax.set_xlabel('block edge length (km)  --  interpolation -> extrapolation')
    ax.set_ylabel('pooled-OOF AUC-PR (positive = Non-abrupt)')
    ax.set_title(f'Repeated block-CV ({N_REPEATS} reshuffles/scale): mean ± 1σ')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)
    plt.close(fig)
    print(f"\nWrote figure: {FIG}")


if __name__ == '__main__':
    main()
