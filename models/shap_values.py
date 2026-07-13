"""Pooled out-of-fold SHAP for the operative thaw-mode model (TASKS T24/T25).

Canonical plumbing (T24): there is NO independent re-split here. The old script did its
own `default_rng(100)` + `train_test_split`, an arbitrary holdout unrelated to the real
CV. This loads `features_clean.csv` with the B6 coordinate quarantine (lat/lon carried
for spatial CV, never in the model matrix) and the persisted CV protocol
(`models/cv_config.json`) — the same spatial-block scheme the trainer uses.

Pooled out-of-fold SHAP (T25): the operative model's SELECTED hyperparameters are held
fixed (read from `models/selected_hparams.json`, the trainer's canonical output). Over
single-level buffered spatial-block folds at the operative cell size, per fold we refit
on the fold-train subset and run TreeSHAP on the HELD-OUT points only, pooling across
folds so every point receives an attribution from a model that never trained on it. The
all-data `model.json` is deliberately not used — OOF attribution requires per-fold refits.

Output space: MARGIN (log-odds), `model_output='raw'` with the exact tree-path-dependent
perturbation (no background dataset). This matches the T19 log-evidence susceptibility
scale. The raw margin of `binary:logistic` is the log-odds of class 1 (Gradual); we
negate so positive SHAP pushes toward Abrupt (class 0), preserving the Abrupt-oriented
sign convention of the earlier figures.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block on plt.show()
import matplotlib.pyplot as plt
import shap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT
from spatial_cv import assign_blocks, buffered_block_folds
# Identical estimator factory + protocol defaults -> parity with training.
from train_xgboost import xgb_builder, OPERATIVE_CELL_KM, BUFFER_KM, N_OUTER, CV_SEED

# Fast smoke config for correctness checks (SHAP_SMOKE=1); does not affect real runs.
SMOKE = bool(os.environ.get('SHAP_SMOKE'))
SMOKE_N = 1500
SMOKE_SPLITS = 3
# Smoke-only fallback hyperparameters (used only if selected_hparams.json is absent AND
# SHAP_SMOKE is set); real runs require the trainer's selected_hparams.json.
SMOKE_HPARAMS = {'max_depth': 3, 'min_child_weight': 20, 'reg_lambda': 10.0,
                 'learning_rate': 0.1, 'n_estimators': 50}

# Dependence plots to emit: (primary feature, interaction feature, output filename).
# Guarded against missing columns so the script survives feature-set changes.
DEPENDENCE_SPECS = [
    ('Slope',                  'Slope',                  'shap_dependence_plot_slope.png'),
    ('Mean curvature (500 m)', 'Mean curvature (500 m)', 'shap_dependence_plot_curvature.png'),
    ('Nitrogen (0-30 cm)',     'Nitrogen (30-200 cm)',   'shap_dependence_plot_nitrogen.png'),
    ('Silt (0-30 cm)',         'Silt (30-200 cm)',       'shap_dependence_plot_sil.png'),
    ('Trend in SWE',           'Trend in SWE',           'shap_dependence_plot_trend_swe.png'),
    ('Mean Annual SWE',        'Mean Annual SWE',        'shap_dependence_plot_mean_annual_swe.png'),
    ('Annual Precipitation',   'Annual Precipitation',   'shap_dependence_plot_annual_precip.png'),
]


# --------------------------------------------------------------------------
# inputs (canonical plumbing — T24)
# --------------------------------------------------------------------------
def load_inputs(feats_csv):
    """Load features with the B6 coordinate quarantine (T7 parity).

    Returns (X, y, lat, lon): X has Class + coords dropped; lat/lon are kept only for
    spatial-block CV, never entering the model matrix.
    """
    feats = pd.read_csv(feats_csv)
    drop = [c for c in ('Class', 'Latitude', 'Longitude') if c in feats.columns]
    X = feats.drop(columns=drop)
    y = feats['Class'].to_numpy()
    lat = feats['Latitude'].to_numpy()
    lon = feats['Longitude'].to_numpy()
    assert 'Latitude' not in X.columns and 'Longitude' not in X.columns, \
        "coordinate quarantine failed: Latitude/Longitude leaked into X"
    return X, y, lat, lon


def load_cv_config(path):
    """Load the persisted CV protocol, resolving missing keys to the trainer defaults.

    An older `cv_config.json` may predate some keys (e.g. `operative_cell_km`); falling
    back to `train_xgboost`'s constants keeps parity and survives a stale config until
    the next training run rewrites it [B6/T11/T24].
    """
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault('operative_cell_km', OPERATIVE_CELL_KM)
    cfg.setdefault('buffer_km', BUFFER_KM)
    cfg.setdefault('n_splits_outer', N_OUTER)
    cfg.setdefault('seeds', {})
    cfg['seeds'].setdefault('CV_SEED', CV_SEED)
    return cfg


def load_selected_hparams(path):
    """Load the operative model's selected hyperparameters [T14/T25].

    `selected_hparams.json` is the trainer's canonical record of the hyperparameters
    the operative model was fit with; OOF SHAP refits every fold with these fixed.
    """
    return json.loads(Path(path).read_text())['hyperparameters']


# --------------------------------------------------------------------------
# pooled out-of-fold SHAP (T25)
# --------------------------------------------------------------------------
def pooled_oof_shap(X, y, lat, lon, *, cell_km, buffer_km, n_splits, seed, hparams):
    """Per-fold refit (fixed hyperparameters) + held-out TreeSHAP, pooled across folds.

    Uses single-level buffered spatial-block folds at `cell_km` (the interpolation scale
    the statewide map serves): each point is held out exactly once, so pooling yields a
    full out-of-fold attribution matrix. SHAP is computed in margin (log-odds) space and
    negated to the Abrupt (class 0) orientation.

    Returns (explanation_abrupt, scored_mask). Points in a fold whose training subset is
    single-class are left unscored (mask False) rather than explained by a degenerate fit.
    """
    y = np.asarray(y)
    n, n_feat = X.shape
    values = np.full((n, n_feat), np.nan)
    base = np.full(n, np.nan)
    scored = np.zeros(n, dtype=bool)

    blocks = assign_blocks(lat, lon, method='albers_grid', cell_km=cell_km)
    folds = buffered_block_folds(lat, lon, blocks, n_splits=n_splits,
                                 buffer_km=buffer_km, seed=seed)
    factory = xgb_builder(hparams)
    for f, (train_idx, test_idx) in enumerate(folds):
        ytr = y[train_idx]
        if len(np.unique(ytr)) < 2 or len(test_idx) == 0:
            continue  # degenerate fold: leave these test points unscored
        est = factory(ytr)
        est.fit(X.iloc[train_idx], ytr)
        # Exact, background-free TreeSHAP in margin (log-odds of class 1 = Gradual).
        explainer = shap.TreeExplainer(est, model_output='raw',
                                       feature_perturbation='tree_path_dependent')
        expl = explainer(X.iloc[test_idx])
        values[test_idx] = expl.values
        base[test_idx] = np.asarray(explainer.expected_value).reshape(-1)[0]
        scored[test_idx] = True

    n_unscored = int((~scored).sum())
    if n_unscored:
        print(f"  [warn] {n_unscored}/{n} points left unscored (single-class fold train); "
              f"excluded from SHAP outputs")

    # Negate margin SHAP -> Abrupt (class 0) orientation: positive pushes toward Abrupt.
    expl_abrupt = shap.Explanation(
        values=-values[scored],
        base_values=-base[scored],
        data=X.values[scored],
        feature_names=list(X.columns),
    )
    return expl_abrupt, scored


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_plots(expl, X_scored, y_scored, out_dir):
    """Emit the SHAP figure set (Abrupt-oriented, margin space) to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(X_scored.columns)
    values = expl.values

    # Dependence plots (skip any whose primary feature is absent).
    for primary, interaction, fname in DEPENDENCE_SPECS:
        if primary not in cols:
            print(f"  [skip] dependence '{primary}': column not present")
            continue
        inter = interaction if interaction in cols else 'auto'
        shap.dependence_plot(primary, values, X_scored, interaction_index=inter, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=300)
        plt.close()

    # Global summary (top features driving Abrupt).
    shap.summary_plot(expl, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary_plot.png', dpi=300)
    plt.close()

    # Beeswarm over actual Abrupt points (class 0): what drives Abrupt predictions.
    abrupt = np.where(y_scored == 0)[0]
    shap.plots.beeswarm(expl[abrupt], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_beeswarm_abrupt.png', dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    cfg = load_cv_config(MODELS / 'cv_config.json')

    hp_path = MODELS / 'selected_hparams.json'
    if hp_path.exists():
        hparams = load_selected_hparams(hp_path)
    elif SMOKE:
        print("[smoke] selected_hparams.json absent; using SMOKE_HPARAMS")
        hparams = SMOKE_HPARAMS
    else:
        raise FileNotFoundError(
            f"{hp_path} not found — run models/train_xgboost.py first so the operative "
            "hyperparameters are recorded (OOF SHAP refits each fold with them).")

    X, y, lat, lon = load_inputs(DATA / 'features_clean.csv')

    n_splits = cfg['n_splits_outer']
    if SMOKE:
        rng = np.random.default_rng(cfg['seeds']['CV_SEED'])
        sel = rng.choice(len(y), size=min(SMOKE_N, len(y)), replace=False)
        X, y, lat, lon = X.iloc[sel].reset_index(drop=True), y[sel], lat[sel], lon[sel]
        n_splits = SMOKE_SPLITS
        print(f"[smoke] subsampled to {len(y)} points, {n_splits} folds")

    print(f"Pooled OOF SHAP: {len(y)} points | {X.shape[1]} features | "
          f"operative cell {cfg['operative_cell_km']} km | buffer {cfg['buffer_km']} km | "
          f"{n_splits} folds | hyperparameters {hparams}")

    expl, scored = pooled_oof_shap(
        X, y, lat, lon,
        cell_km=cfg['operative_cell_km'], buffer_km=cfg['buffer_km'],
        n_splits=n_splits, seed=cfg['seeds']['CV_SEED'], hparams=hparams,
    )

    make_plots(expl, X[scored].reset_index(drop=True), y[scored], OUTPUT)
    print(f"Wrote SHAP figures to {OUTPUT} ({int(scored.sum())} points explained out-of-fold)")


if __name__ == '__main__':
    main()
