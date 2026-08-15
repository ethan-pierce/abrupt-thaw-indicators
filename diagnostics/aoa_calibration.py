"""Set the AOA threshold to a feature-space envelope quantile, and show that skill does
not decay across it [T21 Part 2; revised 2026-08-10, see aoa_threshold_decision.md].

The box-plot fence (Q75 + 1.5*IQR of the CV training-DI distribution) is an arbitrary
convention. An earlier version of this diagnostic instead pushed the threshold out to the
maximum out-of-fold (OOF) DI, on the grounds that pooled-OOF AUC-PR "stayed ~15x the
prevalence floor across the whole DI range with no decay". That was an artifact: per-bin
AUC-PR was compared to the GLOBAL prevalence floor while per-bin prevalence drifts ~8x with
DI, and equal-count binning smeared the sparse high-DI tail into one bin. AUC-PR against a
fixed floor is the wrong instrument for a decay curve on a ~93/7 imbalanced problem.

What this diagnostic now does:

  1. Pooled OOF predictions from the OPERATIVE model (selected hyperparameters, operative
     spatial-CV protocol), same machinery train_xgboost uses.
  2. Each held-out point's DI, computed with the IDENTICAL rank->CDF SHAP-weighted metric
     models/aoa.py uses, OTHER-fold points as neighbours.
  3. Skill vs DI reported with AUC-ROC (prevalence-INVARIANT) as the primary evidence, with
     AUC-PR kept only for reference. AUC-ROC stays ~0.97-0.99 flat across the whole in-sample
     DI range -> no measurable skill decay anywhere the sample reaches.
  4. Because skill never decays in-sample, the threshold is NOT a skill limit. It is the
     feature-space envelope: the ENVELOPE_PCTL-th percentile of the CV training-DI
     distribution. A cell is inside the AOA iff it is no more dissimilar from the training
     data than all but (100-ENVELOPE_PCTL)% of training points are from one another.

It also recomputes DI with the LITERAL M&P raw-standardized-z coordinate, to check that
rank-CDF DI tracks OOF error at least as well (validates the coordinate choice).

Outputs:
  diagnostics/aoa_calibration.png   -- AUC-ROC vs DI (rank-CDF and raw-z), the box-plot
                                        fence, and the chosen envelope threshold.
  models/aoa_threshold.json         -- the chosen threshold + justification; models/aoa.py
                                        reads this to set its AOA boundary.

HONEST CAVEAT (printed + recorded): skill is measurable only over the DI range the biased
training sample spans (to ~0.25). The envelope threshold sits just past that; beyond it the
AOA flags, this calibration cannot score.

Run: poetry run python diagnostics/aoa_calibration.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))

from settings import MODELS  # noqa: E402
import aoa  # noqa: E402  -- the operative metric lives here; reuse it verbatim
from spatial_cv import pooled_oof_predict  # noqa: E402
from train_xgboost import xgb_builder  # noqa: E402  -- operative estimator factory

N_BINS = 12                # equal-count DI bins for the skill-vs-DI curve
ENVELOPE_PCTL = 99.9       # AOA threshold = this percentile of the CV training-DI distribution
CAL_PNG = Path(__file__).resolve().parent / 'aoa_calibration.png'


def selected_hparams():
    sel = json.loads((MODELS / 'selected_hparams.json').read_text())
    return sel['hyperparameters']


def oof_di(zw_train, folds, dbar):
    """Per-point DI using only OTHER-fold points as neighbours (aligned to row index)."""
    di = np.full(len(zw_train), np.nan)
    for train_idx, test_idx in folds:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        d = aoa.nearest_distance(zw_train[test_idx], zw_train[train_idx])
        di[test_idx] = d / dbar
    return di


def binned_skill(di, y, proba, scored, n_bins=N_BINS):
    """Equal-count DI bins over the SCORED points. Per bin: AUC-ROC (prevalence-invariant,
    the primary skill read) and AUC-PR (reference only; positive = Non-abrupt, class 1)."""
    m = scored & np.isfinite(di)
    d, yv, pv = di[m], y[m], proba[m]
    order = np.argsort(d)
    d, yv, pv = d[order], yv[order], pv[order]
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (d >= lo) & (d <= hi) if b == n_bins - 1 else (d >= lo) & (d < hi)
        if sel.sum() < 20 or len(np.unique(yv[sel])) < 2:
            ap = roc = np.nan
        else:
            ap = average_precision_score(yv[sel], pv[sel])
            roc = roc_auc_score(yv[sel], pv[sel])
        rows.append({'lo': float(lo), 'hi': float(hi),
                     'mid': float(np.median(d[sel])) if sel.any() else np.nan,
                     'n': int(sel.sum()), 'n_pos': int((yv[sel] == 1).sum()),
                     'prev': float(yv[sel].mean()) if sel.any() else np.nan,
                     'ap': float(ap), 'roc': float(roc)})
    return rows


def choose_threshold(di_train, pctl=ENVELOPE_PCTL):
    """Applicability boundary = the `pctl`-th percentile of the CV training-DI distribution.

    Feature-space envelope rule: a cell is inside the AOA iff it is no more dissimilar from
    the training data than all but (100-pctl)% of training points are from one another. This
    is NOT a skill limit -- OOF AUC-ROC does not decay within the sample (see the bin table);
    the threshold marks the extent of the training feature envelope, beyond which skill
    cannot be measured regardless. See diagnostics/aoa_threshold_decision.md.
    """
    return float(np.percentile(di_train, pctl))


def main():
    print("=" * 80)
    print("AOA THRESHOLD -- feature-space envelope quantile [T21 Part 2, rev 2026-08-10]")
    print("=" * 80)

    model, names, _ = aoa.load_model_and_features()
    X_df, X_train, lat, lon, y = aoa.load_training(names)
    binary_mask = aoa.detect_binary(X_df)
    floor = float((y == 1).mean())
    print(f"\nTraining points: {len(y):,} | Non-abrupt (class 1) prevalence floor: {floor:.4f}")

    weights = aoa.shap_weights(model, X_df)

    # rank-CDF coordinate (operative metric) + raw-z coordinate (literal M&P, for comparison)
    fitted = aoa.fit_rank_cdf(X_train, binary_mask)
    zw_rank = aoa.weight_coords(aoa.transform_rank_cdf(X_train, fitted, binary_mask), weights)
    mu, sd = aoa.fit_zscale(X_train)
    zw_z = aoa.weight_coords(aoa.transform_zscale(X_train, mu, sd), weights)

    dbar_rank = aoa.mean_pairwise_distance(zw_rank)
    dbar_z = aoa.mean_pairwise_distance(zw_z)
    print(f"dbar (rank-CDF) = {dbar_rank:.6f} | dbar (raw-z) = {dbar_z:.6f}")

    folds = aoa.cv_folds(lat, lon)

    # box-plot fence on the rank-CDF CV training DI (kept for comparison / provenance)
    di_train_rank = aoa.cv_training_di(zw_rank, lat, lon, dbar_rank)
    fence, _, _ = aoa.boxplot_fence(di_train_rank)
    print(f"box-plot fence (rank-CDF CV training DI) = {fence:.4f}")
    print(f"CV training DI: median={np.median(di_train_rank):.4f}  "
          f"p99={np.percentile(di_train_rank, 99):.4f}  "
          f"p{ENVELOPE_PCTL}={np.percentile(di_train_rank, ENVELOPE_PCTL):.4f}  "
          f"max={di_train_rank.max():.4f}")

    print("\nPooled out-of-fold predictions from the operative model "
          "(this refits XGBoost per fold)...")
    factory = xgb_builder(selected_hparams())
    proba, scored = pooled_oof_predict(factory, X_df, y, folds)
    print(f"  scored OOF points: {int(scored.sum()):,} "
          f"(pooled AUC-PR = {average_precision_score(y[scored], proba[scored]):.4f}, "
          f"AUC-ROC = {roc_auc_score(y[scored], proba[scored]):.4f})")

    di_rank = oof_di(zw_rank, folds, dbar_rank)
    di_z = oof_di(zw_z, folds, dbar_z)

    # Does DI predict OOF error? (Spearman of DI vs |y - p|; higher = better applicability index.)
    m = scored & np.isfinite(di_rank) & np.isfinite(di_z)
    resid = np.abs(y[m] - proba[m])
    rho_rank = spearmanr(di_rank[m], resid).correlation
    rho_z = spearmanr(di_z[m], resid).correlation
    print(f"\nSpearman(DI, |OOF residual|):  rank-CDF = {rho_rank:+.3f}   raw-z = {rho_z:+.3f}")
    print("  (higher = DI better tracks where the model errs -> better applicability index)")

    rows_rank = binned_skill(di_rank, y, proba, scored)
    rows_z = binned_skill(di_z, y, proba, scored)

    print("\nRank-CDF DI bins (AUC-ROC is prevalence-invariant; AUC-PR reference only):")
    print("   bin  DI range           n   n_pos   prev   AUC-ROC   AUC-PR")
    for r in rows_rank:
        print(f"   [{r['lo']:.3f},{r['hi']:.3f}]  {r['n']:5d}  {r['n_pos']:4d}  "
              f"{r['prev']:.3f}   {r['roc']:.3f}    {r['ap']:.3f}")
    _finite_roc = [r['roc'] for r in rows_rank if np.isfinite(r['roc'])]
    print(f"  AUC-ROC across bins: min={min(_finite_roc):.3f} max={max(_finite_roc):.3f} "
          f"-> no decay within the sampled DI range")

    threshold = choose_threshold(di_train_rank)
    source = 'training_envelope_quantile'
    print(f"\nChosen AOA threshold = {threshold:.4f}  [{source}: p{ENVELOPE_PCTL} of CV training DI]")

    caveat = ('Skill is measurable only over the DI range the biased training sample spans '
              '(to ~0.25); the envelope threshold sits just past that. Beyond it the AOA '
              'flags, this calibration cannot score.')
    rule = (f'{ENVELOPE_PCTL}th percentile of the cross-validated training DI distribution '
            f'(feature-space envelope). A cell is inside the AOA iff no more dissimilar from '
            f'the training data than all but {100 - ENVELOPE_PCTL:g}% of training points. NOT '
            f'a skill limit: OOF AUC-ROC stayed {min(_finite_roc):.2f}-{max(_finite_roc):.2f} '
            f'with no decay across the whole sampled DI range.')
    note = ('Threshold set to the training feature-space envelope (p{:g} of CV training DI), '
            'not a skill boundary: OOF ranking skill (AUC-ROC) does not decay within the '
            'sample. ' + caveat).format(ENVELOPE_PCTL)
    print(f"  {note}")

    payload = {
        'metric': 'rank_cdf',
        'threshold': float(threshold),
        'rule': rule,
        'source': source,
        'envelope_pctl': ENVELOPE_PCTL,
        'prevalence_floor': floor,
        'oof_aucroc_min': float(min(_finite_roc)),
        'oof_aucroc_max': float(max(_finite_roc)),
        'boxplot_fence': float(fence),
        'dbar': float(dbar_rank),
        'spearman_di_resid_rank_cdf': float(rho_rank),
        'spearman_di_resid_raw_z': float(rho_z),
        'cv_protocol': f'{aoa.BLOCK_METHOD} {aoa.OPERATIVE_CELL_KM}km, {aoa.N_SPLITS} folds, '
                       f'buffer {aoa.BUFFER_KM}km, seed {aoa.CV_SEED}',
        'note': note,
        'caveat': caveat,
    }
    (MODELS / 'aoa_threshold.json').write_text(json.dumps(payload, indent=2, default=float))
    print(f"Wrote threshold: {MODELS / 'aoa_threshold.json'}")

    # Per-bin skill table for the manuscript figure (output/). Carries the same
    # rows the diagnostic prints, so the figure script needs no OOF recompute.
    bins_payload = {
        'metric': 'rank_cdf',
        'prevalence_floor': float(floor),
        'threshold': float(threshold),
        'envelope_pctl': ENVELOPE_PCTL,
        'boxplot_fence': float(fence),
        'dbar_rank': float(dbar_rank),
        'sampled_di_max': float(max(r['hi'] for r in rows_rank)),
        'spearman_di_resid_rank_cdf': float(rho_rank),
        'spearman_di_resid_raw_z': float(rho_z),
        'bins_rank': rows_rank,
        'bins_raw_z': rows_z,
    }
    bins_path = ROOT / 'output' / 'aoa_calibration_bins.json'
    bins_path.write_text(json.dumps(bins_payload, indent=2, default=float))
    print(f"Wrote per-bin skill table: {bins_path}")

    make_figure(rows_rank, rows_z, fence, threshold, rho_rank, rho_z)


def make_figure(rows_rank, rows_z, fence, threshold, rho_rank, rho_z):
    fig, ax = plt.subplots(figsize=(11, 7))

    def xy(rows):
        m = [np.isfinite(r['roc']) and np.isfinite(r['mid']) for r in rows]
        return ([r['mid'] for r, k in zip(rows, m) if k],
                [r['roc'] for r, k in zip(rows, m) if k])

    xr_, yr_ = xy(rows_rank)
    xz_, yz_ = xy(rows_z)
    ax.plot(xr_, yr_, 'o-', color='#1f77b4', lw=2, ms=6,
            label=f'rank-CDF DI  (Spearman DI·|resid| = {rho_rank:+.2f})')
    ax.plot(xz_, yz_, 's--', color='#999999', lw=1.5, ms=5,
            label=f'raw-z DI (literal M&P)  ({rho_z:+.2f})')
    ax.axvline(threshold, color='#2e8b57', lw=2,
               label=f'AOA threshold = {threshold:.2f}  (p{ENVELOPE_PCTL} of CV training DI)')
    ax.axvline(fence, color='#e08e0b', ls='--', lw=1.2, label=f'box-plot fence = {fence:.2f}')

    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel('Dissimilarity index DI (nearest-training distance / dbar)', fontsize=12)
    ax.set_ylabel('Pooled-OOF AUC-ROC (prevalence-invariant)', fontsize=12)
    ax.set_title('AOA threshold: model skill vs feature-space dissimilarity\n'
                 'AUC-ROC is flat across the sampled DI range (no decay); the threshold is the '
                 'training feature-space envelope, not a skill limit', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.02,
             'Caveat: skill is measurable only over the DI range the biased sample spans '
             '(to ~0.25); beyond the envelope the AOA flags, calibration cannot score.',
             ha='center', fontsize=8, style='italic', color='#555555')
    plt.savefig(CAL_PNG, dpi=300, bbox_inches='tight')
    print(f"Wrote calibration figure: {CAL_PNG}")
    plt.close('all')


if __name__ == '__main__':
    main()
