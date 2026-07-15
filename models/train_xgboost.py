"""Nested spatial-CV training & hyperparameter selection for the thaw-mode model.

Rewritten for the methods-cleanup pipeline (TASKS T8-T12). The leaky random
`train_test_split` + `StratifiedKFold` `GridSearchCV` is replaced by NESTED buffered
spatial-block CV (see `spatial_cv.py`) run across a sweep of block sizes:

  outer folds  -> headline: pooled out-of-fold AUC-PR + across-fold spread   [B4/D11]
  inner folds  -> hyperparameter selection on pooled-OOF AUC-PR              [C8]
  block size   -> inference regime: small = interpolation, large = extrapolation [B5a]

Design choices tied to the likelihood-ratio framing (E13):
  * `scale_pos_weight = 1` (no imbalance reweighting) so the divided-out prior stays
    exactly the sample prevalence.                                          [C9/T10]
  * Selection & headline are AUC-PR (positive = Gradual, class 1); accuracy is not
    reported at all (meaningless at ~93% prevalence).                       [D11/T12]

This script produces the CV evidence (sweep curve + config). The single operative
`model.json` is the all-data refit with the selected hyperparameters -> T14 (not here).
"""

import os
import json
import hashlib
import itertools
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block on plt.show()
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import ROOT, DATA, MODELS, OUTPUT
from spatial_cv import (assign_blocks, nested_block_folds, buffered_block_folds,
                        pooled_oof_predict)

# --------------------------------------------------------------------------
# config — named seeds (42 lineage) and the CV protocol [C10/T11]
# --------------------------------------------------------------------------
SPLIT_SEED = 42   # retained for lineage; the random holdout split is retired (B4)
MODEL_SEED = 42   # XGBoost estimator randomness
CV_SEED = 42      # block-fold shuffling (deterministic fold regeneration)

BLOCK_METHOD = 'albers_grid'          # equal-area km grid [B5b]
BUFFER_KM = 0.0                        # empirical: block holdout already removes near-seam leakage [T43]
N_OUTER = 5
N_INNER = 5
SWEEP_CELL_KM = [10, 25, 50, 100, 200]  # interpolation -> extrapolation [B5a]
OPERATIVE_CELL_KM = 10  # scale for selecting the operative model's hyperparameters (T14)

# Grid widened on principled axes, small enough for nested CV [C10/T11].
PARAM_GRID = {
    'max_depth':        [3, 5],
    'min_child_weight': [5, 20],
    'reg_lambda':       [1.0, 10.0],
    'learning_rate':    [0.05, 0.1],
    'n_estimators':     [200, 400],
}
# Penalized-logistic baseline regularization grid [D12/T13].
LOGIT_GRID = {'C': [0.01, 0.1, 1.0]}

# Heavy-tailed non-negative features the LINEAR baseline log-compresses in its own
# Pipeline [T35 bucket-3 / T45]. XGBoost is scale-invariant and sees these raw; the
# canonical table stays raw. Un-logged, their orders-of-magnitude tails (Upstream Area
# spans river-basin scales) blow up StandardScaler + the lbfgs matmul on the finite
# inputs. Precipitation *amounts* only — Precipitation Seasonality is a bounded CV.
LOG_BASELINE_COLS = (
    'Height Above Nearest Drainage',
    'Upstream Area',
    'Annual Precipitation',
    'Precipitation of Wettest Month',
    'Precipitation of Driest Month',
    'Precipitation of Wettest Quarter',
    'Precipitation of Driest Quarter',
    'Precipitation of Warmest Quarter',
    'Precipitation of Coldest Quarter',
    'Mean Annual SWE',
    'Soil Organic Carbon (0-30 cm)',
    'Soil Organic Carbon (30-200 cm)',
    'Nitrogen (0-30 cm)',
    'Nitrogen (30-200 cm)',
)


def _log1p_nonneg(a):
    """log1p with negatives clipped to 0 (defensive) and NaN preserved for downstream
    median-imputation — so the transform never emits invalid-value/overflow warnings."""
    return np.log1p(np.clip(a, 0.0, None))


def _binary_cols(X):
    """Columns whose (non-NaN) values are all in {0, 1} — the one-hot Land Cover /
    Vegetation Mode indicators and Yedoma. Detected by value (not name) so it survives
    renames. These are passed to the baseline WITHOUT standardization (see below)."""
    out = []
    for c in X.columns:
        u = pd.unique(X[c].dropna())
        if len(u) and set(np.asarray(u, dtype=float).tolist()) <= {0.0, 1.0}:
            out.append(c)
    return out


def _log_cont_cols(X):
    """Continuous columns that get log-compressed (heavy-tailed, non-binary)."""
    binary = set(_binary_cols(X))
    return [c for c in X.columns if c in LOG_BASELINE_COLS and c not in binary]


def _other_cont_cols(X):
    """Continuous columns standardized without a log (everything not log/not binary)."""
    binary = set(_binary_cols(X))
    return [c for c in X.columns if c not in LOG_BASELINE_COLS and c not in binary]


# Fast smoke config for correctness checks (TRAIN_SMOKE=1); does not affect real runs.
SMOKE = bool(os.environ.get('TRAIN_SMOKE'))
if SMOKE:
    SWEEP_CELL_KM = [50, 200]
    N_OUTER, N_INNER = 3, 3
    PARAM_GRID = {'max_depth': [3, 5], 'min_child_weight': [20],
                  'reg_lambda': [10.0], 'learning_rate': [0.1], 'n_estimators': [100]}
    LOGIT_GRID = {'C': [0.1, 1.0]}


# --------------------------------------------------------------------------
# estimator + grid helpers
# --------------------------------------------------------------------------
def xgb_builder(params):
    """`builder(params) -> factory(y_train) -> fresh XGBClassifier`.

    `scale_pos_weight = 1` (T10): no imbalance reweighting. Factories take `y_train`
    to satisfy the `pooled_oof_predict` interface but ignore it (no reweighting).
    """
    def factory(y_train):
        return xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            tree_method='hist',
            scale_pos_weight=1,
            random_state=MODEL_SEED,
            n_jobs=4,
            **params,
        )
    return factory


class _QuietLinearBaseline:
    """fit/predict_proba wrapper scoping `np.errstate` around the L2-logistic baseline.

    liblinear's predict-time decision-function matmul trips numpy's FP sticky-flag
    reporting even on finite inputs — a known benign numpy SIMD false positive (the
    decision values are finite; verified T45). Ignoring divide/over/invalid here keeps
    that noise off stderr WITHOUT masking any real non-finite value, and is scoped to the
    baseline only (XGBoost is untouched). Implements just the `pooled_oof_predict`
    interface (`fit`, `predict_proba`)."""
    _ERR = dict(divide='ignore', over='ignore', invalid='ignore')

    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y):
        with np.errstate(**self._ERR):
            self.pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        with np.errstate(**self._ERR):
            return self.pipe.predict_proba(X)


def logistic_builder(params):
    """Penalized-logistic baseline; the linear model owns its preprocessing [D12/T35/T45].

    Preprocessing is split by column type so the solver sees well-conditioned inputs
    (raw un-conditioned inputs flood lbfgs with overflow/invalid-matmul warnings — T45):
      - heavy-tailed non-negative continuous (`LOG_BASELINE_COLS`): log1p -> median-impute
        -> standardize (their orders-of-magnitude tails, e.g. Upstream Area, otherwise
        dominate the scale);
      - other continuous: median-impute -> standardize;
      - binary one-hot indicators (Land Cover / Vegetation Mode / Yedoma): **not
        standardized** — dividing a rare 0/1 column by its tiny std inflates its lone `1`
        to ~100+ sigma, and on near-separable data lbfgs transiently overflows on it.
        Missing filled with 0 (absent category).
    `class_weight=None` mirrors the no-reweighting choice (C9). Every step is fit inside
    each fold's pipeline (via `pooled_oof_predict`), so nothing leaks across the CV split.
    """
    prep = ColumnTransformer(
        [
            ('log_cont', Pipeline([('log', FunctionTransformer(_log1p_nonneg)),
                                   ('impute', SimpleImputer(strategy='median')),
                                   ('scale', StandardScaler())]), _log_cont_cols),
            ('cont', Pipeline([('impute', SimpleImputer(strategy='median')),
                               ('scale', StandardScaler())]), _other_cont_cols),
            ('binary', SimpleImputer(strategy='constant', fill_value=0.0), _binary_cols),
        ],
        remainder='drop',
    )

    def factory(y_train):
        pipe = Pipeline([
            ('prep', prep),
            # solver='liblinear' (coordinate descent) not lbfgs: on this near-separable
            # data lbfgs floods stderr with transient overflow/invalid-matmul warnings
            # over its ~100 line-search iterations (the final coef is small & finite —
            # the warnings are benign optimizer noise). liblinear reaches the identical
            # fit in <10 iterations, warning-free (T45).
            ('clf', LogisticRegression(penalty='l2', C=params['C'], class_weight=None,
                                       solver='liblinear', max_iter=1000,
                                       random_state=MODEL_SEED)),
        ])
        return _QuietLinearBaseline(pipe)
    return factory


def dummy_builder(strategy):
    """No-skill baseline family; `strategy` in {'prior','stratified'} [D12]."""
    def builder(params):  # params ignored (no hyperparameters)
        def factory(y_train):
            return DummyClassifier(strategy=strategy, random_state=MODEL_SEED)
        return factory
    return builder


# Estimator families run through the IDENTICAL nested folds: (name, builder, grid).
def make_families():
    return [
        ('xgboost', xgb_builder, PARAM_GRID),
        ('logistic', logistic_builder, LOGIT_GRID),
        ('dummy_prior', dummy_builder('prior'), {}),
        ('dummy_stratified', dummy_builder('stratified'), {}),
    ]


def grid_combos(param_grid):
    """Yield every hyperparameter combination as a dict (Cartesian product)."""
    keys = list(param_grid)
    for values in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))


def _safe_ap(y_true, proba, mask):
    """Average precision on the scored, both-class subset; NaN if undefined."""
    if mask.sum() == 0:
        return np.nan
    yt, pp = y_true[mask], proba[mask]
    if len(np.unique(yt)) < 2:
        return np.nan
    return average_precision_score(yt, pp)


# --------------------------------------------------------------------------
# selection + sweep
# --------------------------------------------------------------------------
def select_hparams(X, y, inner_folds, builder, param_grid):
    """Pick the grid combo maximizing pooled-OOF AUC-PR over the inner folds [C8/T9]."""
    yv = np.asarray(y)
    best_combo, best_ap = None, -np.inf
    for combo in grid_combos(param_grid):
        proba, scored = pooled_oof_predict(builder(combo), X, y, inner_folds)
        ap = _safe_ap(yv, proba, scored)
        if np.isfinite(ap) and ap > best_ap:
            best_ap, best_combo = ap, combo
    if best_combo is None:  # extreme minority sparsity: fall back to a sane default
        best_combo = next(grid_combos(param_grid))
    return best_combo, best_ap


def run_family(X, y, outer_folds, builder, param_grid):
    """One estimator family through the (pre-materialized) nested folds -> diagnostics."""
    yv = np.asarray(y)
    oof_proba = np.full(len(yv), np.nan)
    oof_scored = np.zeros(len(yv), dtype=bool)
    per_fold = []
    for f, (otr, ote, inner) in enumerate(outer_folds):
        combo, inner_ap = select_hparams(X, y, inner, builder, param_grid)
        est = builder(combo)(yv[otr])
        est.fit(X.iloc[otr], yv[otr])
        oof_proba[ote] = est.predict_proba(X.iloc[ote])[:, 1]
        oof_scored[ote] = True
        fold_mask = np.zeros(len(yv), dtype=bool)
        fold_mask[ote] = True
        per_fold.append({
            'fold': f, 'inner_ap': inner_ap, 'fold_ap': _safe_ap(yv, oof_proba, fold_mask),
            'selected': combo,
            'test_n': int(len(ote)), 'test_gradual': int((yv[ote] == 1).sum()),
            'train_n': int(len(otr)), 'train_gradual': int((yv[otr] == 1).sum()),
        })

    pooled_ap = _safe_ap(yv, oof_proba, oof_scored)
    pooled_roc = (roc_auc_score(yv[oof_scored], oof_proba[oof_scored])
                  if oof_scored.sum() and len(np.unique(yv[oof_scored])) == 2 else np.nan)
    fold_aps = np.array([r['fold_ap'] for r in per_fold if np.isfinite(r['fold_ap'])])
    return {
        'pooled_ap': pooled_ap,
        'pooled_roc': pooled_roc,
        'fold_ap_mean': float(np.mean(fold_aps)) if len(fold_aps) else np.nan,
        'fold_ap_std': float(np.std(fold_aps)) if len(fold_aps) else np.nan,
        'fold_ap_min': float(np.min(fold_aps)) if len(fold_aps) else np.nan,
        'fold_ap_max': float(np.max(fold_aps)) if len(fold_aps) else np.nan,
        'n_scored': int(oof_scored.sum()),
        'per_fold': per_fold,
    }


def run_block_size(X, y, lat, lon, cell_km, families):
    """Nested spatial CV at one block size, all families through the SAME folds [T8/T13]."""
    blocks = assign_blocks(lat, lon, method=BLOCK_METHOD, cell_km=cell_km)
    # Materialize once so every family sees identical outer/inner folds.
    outer_folds = list(nested_block_folds(lat, lon, blocks, n_splits_outer=N_OUTER,
                                          n_splits_inner=N_INNER, buffer_km=BUFFER_KM,
                                          seed=CV_SEED))
    families_res = {name: run_family(X, y, outer_folds, builder, grid)
                    for name, builder, grid in families}
    return {'cell_km': cell_km, 'families': families_res}


# --------------------------------------------------------------------------
# reproducibility check + persistence
# --------------------------------------------------------------------------
def assert_folds_reproducible(lat, lon):
    """Regenerating folds from the same seeds yields identical splits [T11]."""
    blocks = assign_blocks(lat, lon, method=BLOCK_METHOD, cell_km=SWEEP_CELL_KM[0])
    def signature():
        return [(tuple(otr.tolist()), tuple(ote.tolist()))
                for otr, ote, _ in nested_block_folds(
                    lat, lon, blocks, n_splits_outer=N_OUTER, n_splits_inner=N_INNER,
                    buffer_km=BUFFER_KM, seed=CV_SEED)]
    assert signature() == signature(), "folds are not reproducible from the CV config"


def cv_config_dict():
    """The CV protocol + seeds; single source of truth for config + manifest [B6/T11]."""
    return {
        'block_method': BLOCK_METHOD,
        'sweep_cell_km': SWEEP_CELL_KM,
        'operative_cell_km': OPERATIVE_CELL_KM,
        'buffer_km': BUFFER_KM,
        'n_splits_outer': N_OUTER,
        'n_splits_inner': N_INNER,
        'seeds': {'SPLIT_SEED': SPLIT_SEED, 'MODEL_SEED': MODEL_SEED, 'CV_SEED': CV_SEED},
        'param_grid': PARAM_GRID,
        'logit_grid': LOGIT_GRID,
        'smoke': SMOKE,
    }


def write_cv_config():
    """Persist the CV config + seeds for deterministic fold regeneration [B6/T11]."""
    path = MODELS / 'cv_config.json'
    path.write_text(json.dumps(cv_config_dict(), indent=2))
    return path


def _git_info():
    """Current git SHA + dirty flag; None if unavailable (not fatal to a run)."""
    def run(*args):
        return subprocess.check_output(args, cwd=str(ROOT), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return {'sha': run('git', 'rev-parse', 'HEAD'),
                'dirty': bool(run('git', 'status', '--porcelain'))}
    except Exception as e:
        return {'sha': None, 'dirty': None, 'error': str(e)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def _product_versions():
    """Identify the upstream permafrost products by their versioned filenames [H20.1]."""
    def names(pattern):
        return sorted(p.name for p in DATA.glob(pattern))
    obu = sorted(set(names('*PERPROB*') + names('Obu*')))
    brown = 'arctic-permafrost-map' if (DATA / 'arctic-permafrost-map').is_dir() else None
    thawdb = names('Alaska_Permafrost_Thaw_Database_v*.csv')
    return {'obu': obu, 'brown_ipa': brown, 'thawdb': thawdb}


def write_run_manifest(selected, path=None):
    """Write the reproducibility manifest beside model.json, each run [H20.1/T16]."""
    path = (MODELS / 'run_manifest.json') if path is None else Path(path)
    feats_csv = DATA / 'features_clean.csv'
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git': _git_info(),
        'features_clean_csv': {
            'path': str(feats_csv),
            'sha256': _sha256(feats_csv) if feats_csv.exists() else None,
        },
        'cv_config': cv_config_dict(),
        'product_versions': _product_versions(),
        'selected_hyperparameters': selected,
    }
    path.write_text(json.dumps(manifest, indent=2, default=float))
    return path


def refit_operative_model(X, y, lat, lon, model_path=None, cell_km=None):
    """Select operative hyperparameters, refit on ALL data, save model.json [B6/T14].

    Selection uses single-level buffered block CV over all data at `cell_km` (the
    interpolation scale that the statewide map serves), maximizing pooled-OOF AUC-PR.
    This is not double-dipping: the honest performance estimate is the nested sweep
    (T8); this only picks the final hyperparameters. The refit is on every row, and
    `save_model` embeds `learner.feature_names` for predict.py / shap_values.py.
    """
    cell_km = OPERATIVE_CELL_KM if cell_km is None else cell_km
    model_path = (MODELS / 'model.json') if model_path is None else Path(model_path)
    yv = np.asarray(y)
    blocks = assign_blocks(lat, lon, method=BLOCK_METHOD, cell_km=cell_km)
    folds = list(buffered_block_folds(lat, lon, blocks, n_splits=N_OUTER,
                                      buffer_km=BUFFER_KM, seed=CV_SEED))
    combo, ap = select_hparams(X, y, folds, xgb_builder, PARAM_GRID)

    est = xgb_builder(combo)(yv)
    est.fit(X, yv)  # all data; DataFrame X -> feature names embedded on save
    est.save_model(str(model_path))

    sel = {'operative_cell_km': cell_km, 'selection_auc_pr': ap,
           'n_train': int(len(yv)), 'hyperparameters': combo}
    (model_path.parent / 'selected_hparams.json').write_text(
        json.dumps(sel, indent=2, default=float))
    return combo, ap


def plot_sweep_curve(results, family_names, prevalence, path):
    """AUC-PR vs block size per family, with spread + prevalence floor [D11/T12/T13]."""
    cells = [r['cell_km'] for r in results]
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in family_names:
        means = [r['families'][name]['fold_ap_mean'] for r in results]
        stds = [r['families'][name]['fold_ap_std'] for r in results]
        if name == 'xgboost':
            ax.errorbar(cells, means, yerr=stds, marker='o', capsize=4, linewidth=2,
                        label='xgboost (mean +/- std across folds)')
        else:
            ax.plot(cells, means, marker='.', linestyle='--', alpha=0.8, label=name)
    ax.axhline(prevalence, color='grey', linestyle=':',
               label=f'prevalence floor ({prevalence:.3f})')
    ax.set_xscale('log')
    ax.set_xlabel('block edge length (km)  --  interpolation -> extrapolation')
    ax.set_ylabel('AUC-PR (positive = Gradual)')
    ax.set_title('Nested spatial-CV AUC-PR vs block size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    feats = pd.read_csv(DATA / 'features_clean.csv')

    # Quarantine coordinates (B6/T7): carried for spatial CV, never in the model matrix.
    X = feats.drop(['Class', 'Latitude', 'Longitude'], axis=1)
    y = feats['Class']
    coords = feats[['Latitude', 'Longitude']]
    assert 'Latitude' not in X.columns and 'Longitude' not in X.columns, \
        "coordinate quarantine failed: Latitude/Longitude leaked into X"
    missing_log = [c for c in LOG_BASELINE_COLS if c not in X.columns]
    assert not missing_log, \
        f"baseline log columns missing from feature table (renamed?): {missing_log}"

    lat = coords['Latitude'].to_numpy()
    lon = coords['Longitude'].to_numpy()
    prevalence = float((y == 1).mean())  # baseline AUC-PR floor (Gradual fraction)

    print(f"Samples: {len(y)} | features: {X.shape[1]} | "
          f"Gradual prevalence (AUC-PR floor): {prevalence:.4f}")
    print(f"Block method: {BLOCK_METHOD} | buffer: {BUFFER_KM} km | "
          f"outer/inner folds: {N_OUTER}/{N_INNER} | seeds: "
          f"SPLIT={SPLIT_SEED} MODEL={MODEL_SEED} CV={CV_SEED}")

    assert_folds_reproducible(lat, lon)
    cfg_path = write_cv_config()
    print(f"Wrote CV config: {cfg_path}")

    families = make_families()
    family_names = [name for name, _, _ in families]

    results = []
    for cell_km in SWEEP_CELL_KM:
        print(f"\n=== block size {cell_km} km ===")
        res = run_block_size(X, y, lat, lon, cell_km, families)
        results.append(res)
        xg = res['families']['xgboost']
        print(f"xgboost  pooled-OOF AUC-PR: {xg['pooled_ap']:.4f} | "
              f"across-fold: {xg['fold_ap_mean']:.4f} +/- {xg['fold_ap_std']:.4f} "
              f"[{xg['fold_ap_min']:.4f}, {xg['fold_ap_max']:.4f}] | "
              f"AUC-ROC (secondary): {xg['pooled_roc']:.4f} | scored: {xg['n_scored']}")
        print("  baselines (pooled-OOF AUC-PR):", ", ".join(
            f"{name}={res['families'][name]['pooled_ap']:.4f}"
            for name in family_names if name != 'xgboost'))
        for r in xg['per_fold']:
            print(f"  xgb fold {r['fold']}: test Gradual {r['test_gradual']}/{r['test_n']}, "
                  f"train Gradual {r['train_gradual']}/{r['train_n']}, "
                  f"fold AUC-PR {r['fold_ap']:.4f}")

    # Headline curve + machine-readable sweep results [D11/T12/T13].
    curve_path = OUTPUT / 'aucpr_vs_blocksize.png'
    plot_sweep_curve(results, family_names, prevalence, curve_path)
    print(f"\nWrote AUC-PR vs block-size curve: {curve_path}")

    summary = {'prevalence_floor': prevalence, 'results': results}
    res_path = OUTPUT / 'cv_sweep_results.json'
    res_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote sweep results: {res_path}")

    # Operative model: select hyperparameters + refit on ALL data -> model.json [T14].
    combo, ap = refit_operative_model(X, y, lat, lon)
    print(f"\nOperative model refit on all {len(y)} rows "
          f"(selection AUC-PR {ap:.4f} @ {OPERATIVE_CELL_KM} km):")
    print(f"  selected hyperparameters: {combo}")
    print(f"  saved: {MODELS / 'model.json'} and {MODELS / 'selected_hparams.json'}")

    # Reproducibility manifest beside model.json [H20.1/T16].
    selected = {'hyperparameters': combo, 'selection_auc_pr': ap,
                'operative_cell_km': OPERATIVE_CELL_KM}
    manifest_path = write_run_manifest(selected)
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == '__main__':
    main()
