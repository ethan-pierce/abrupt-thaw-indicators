"""Per-feature OOF SHAP cache for Figure 8 (mechanism / dependence shapes).

Figure 8 needs per-FEATURE SHAP paired with the underlying feature VALUES, so it
can draw family-summed dependence shapes (family-sum SHAP on y vs. the family's
leading member's value on x) and decompose the Land Cover family per class.

The grouped-family cache (output/shap_grouped_matrix.npz, from shap_groups.py)
keeps only the family-SUMMED SHAP — it discards the per-column values and the
feature data. So this script re-runs the SAME canonical OOF machinery
(pooled_oof_shap: per-fold refit + held-out TreeSHAP, Abrupt-oriented margin)
and persists the full per-feature arrays instead.

Consistency guarantees (so Fig 8 lines up with Fig 7):
  * Same inputs (data/features_clean.csv), same CV config + selected hparams,
    same seed  -> identical `scored` set and identical SHAP values.
  * Family memberships are NOT recomputed here; Fig 8 reads them from
    output/shap_families.json (written by shap_groups.py), the single source of
    truth. This cache only supplies per-feature (values, data) so the family sums
    it forms downstream equal the columns of shap_grouped_matrix.npz exactly.

Run after the operative model / feature set is final:
    poetry run python models/shap_mechanism_cache.py
Writes output/shap_mechanism_cache.npz. Multi-minute run (fold-refit TreeSHAP).
SHAP_MECH_SMOKE=1 subsamples for a fast wiring check (writes to output/_smoke/).
"""

import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT
from shap_values import (load_inputs, load_cv_config, load_selected_hparams,
                         pooled_oof_shap, SMOKE_HPARAMS)

SMOKE = bool(os.environ.get('SHAP_MECH_SMOKE'))
SMOKE_N = 1500
SMOKE_SPLITS = 3


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
            f"{hp_path} not found -- run models/train_xgboost.py first so the operative "
            "hyperparameters are recorded (OOF SHAP refits each fold with them).")

    X, y, lat, lon = load_inputs(DATA / 'features_clean.csv')

    n_splits = cfg['n_splits_outer']
    if SMOKE:
        rng = np.random.default_rng(cfg['seeds']['CV_SEED'])
        sel = rng.choice(len(y), size=min(SMOKE_N, len(y)), replace=False)
        X, y, lat, lon = X.iloc[sel].reset_index(drop=True), y[sel], lat[sel], lon[sel]
        n_splits = SMOKE_SPLITS
        print(f"[smoke] subsampled to {len(y)} points, {n_splits} folds")

    print(f"Per-feature OOF SHAP: {len(y)} points | {X.shape[1]} features | "
          f"operative cell {cfg['operative_cell_km']} km | buffer {cfg['buffer_km']} km | "
          f"{n_splits} folds | hyperparameters {hparams}")

    expl, scored = pooled_oof_shap(
        X, y, lat, lon,
        cell_km=cfg['operative_cell_km'], buffer_km=cfg['buffer_km'],
        n_splits=n_splits, seed=cfg['seeds']['CV_SEED'], hparams=hparams,
    )

    out_dir = OUTPUT / '_smoke' if SMOKE else OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'shap_mechanism_cache.npz'
    np.savez(
        out,
        values=expl.values.astype(np.float32),          # (n_scored, F) per-feature SHAP, Abrupt-oriented
        data=np.asarray(expl.data, dtype=np.float32),   # (n_scored, F) feature values
        feature_names=np.array(list(expl.feature_names), dtype=object),
        y=y[scored].astype(np.int8),                    # labels for the scored points (0=Abrupt,1=Non-abrupt)
        smoke=np.array(SMOKE),
    )
    print(f"\nWrote {out}  "
          f"[values {expl.values.shape}, {int(scored.sum())} points explained out-of-fold]")
    print("Family memberships are NOT stored here — Fig 8 reads them from "
          "output/shap_families.json (source of truth); this cache supplies only "
          "per-feature (values, data) so family sums match shap_grouped_matrix.npz.")


if __name__ == '__main__':
    main()
