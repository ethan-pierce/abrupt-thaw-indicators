"""Per-feature OOF SHAP cache for Figure 7 (mechanism / dependence shapes).

Figure 7 needs per-feature SHAP paired with the underlying feature values, to draw
family-summed dependence shapes and decompose the Land Cover family per class. The
grouped-family cache (output/shap_grouped_matrix.npz) keeps only family-summed SHAP,
so this script re-runs the same OOF machinery (pooled_oof_shap) and persists the full
per-feature arrays.

Same inputs, CV config, hyperparameters, and seed as shap_groups.py, so the family
sums formed downstream equal the columns of shap_grouped_matrix.npz. Family
memberships are not stored here; Fig 7 reads them from output/shap_families.json.

    poetry run python models/shap_mechanism_cache.py

Writes output/shap_mechanism_cache.npz (multi-minute).
"""

from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT
from shap_values import (load_inputs, load_cv_config, load_selected_hparams,
                         pooled_oof_shap)


def main():
    cfg = load_cv_config(MODELS / 'cv_config.json')

    hp_path = MODELS / 'selected_hparams.json'
    if not hp_path.exists():
        raise FileNotFoundError(
            f"{hp_path} not found -- run models/train_xgboost.py first so the operative "
            "hyperparameters are recorded (OOF SHAP refits each fold with them).")
    hparams = load_selected_hparams(hp_path)

    X, y, lat, lon = load_inputs(DATA / 'features_clean.csv')

    n_splits = cfg['n_splits_outer']

    print(f"Per-feature OOF SHAP: {len(y)} points | {X.shape[1]} features | "
          f"operative cell {cfg['operative_cell_km']} km | buffer {cfg['buffer_km']} km | "
          f"{n_splits} folds | hyperparameters {hparams}")

    expl, scored = pooled_oof_shap(
        X, y, lat, lon,
        cell_km=cfg['operative_cell_km'], buffer_km=cfg['buffer_km'],
        n_splits=n_splits, seed=cfg['seeds']['CV_SEED'], hparams=hparams,
    )

    out = OUTPUT / 'shap_mechanism_cache.npz'
    np.savez(
        out,
        values=expl.values.astype(np.float32),          # (n_scored, F) per-feature SHAP, Abrupt-oriented
        data=np.asarray(expl.data, dtype=np.float32),   # (n_scored, F) feature values
        feature_names=np.array(list(expl.feature_names), dtype=object),
        y=y[scored].astype(np.int8),                    # labels for the scored points (0=Abrupt,1=Non-abrupt)
    )
    print(f"\nWrote {out}  "
          f"[values {expl.values.shape}, {int(scored.sum())} points explained out-of-fold]")
    print("Family memberships are NOT stored here — Fig 7 reads them from "
          "output/shap_families.json (source of truth); this cache supplies only "
          "per-feature (values, data) so family sums match shap_grouped_matrix.npz.")


if __name__ == '__main__':
    main()
