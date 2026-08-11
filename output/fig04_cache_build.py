"""Compute + cache the arrays for Figure 4 (redesigned two-panel).

Panel (a) — pooled out-of-fold precision-recall CURVE at the operative 10 km block
scale. For each of N_REPEATS block->fold reshuffles we pool the OOF predictions and
build one PR curve per partition (XGBoost operative hparams + logistic baseline),
then interpolate every curve onto a shared recall grid. The mean curve is drawn with
a +/-1 sigma ACROSS-PARTITION band -- the same partition-robustness uncertainty the
repeated-CV AUC-PR (0.852 +/- 0.011) reports. Fold-to-fold / spatial heterogeneity is
NOT put here; panel (b) shows it as skill-vs-distance, so the two panels never
double-count the same uncertainty.

Panel (b) — AUC-PR vs median great-circle distance to the nearest training point, on
ONE shared axis for both spatial-holdout geometries: the block-CV sweep (square
albers tiles, 5..200 km) and the leave-region-out sweep (contiguous k-means clusters,
50..3 regions). Block AUC-PR comes from repeated_cv_results.json; the region series
(distance + AUC-PR) from extrapolation_range_results.json. Only the block-CV median
distances are computed here (they were never cached), from the seed-42 partition.

Writes output/fig04_cache.npz. Run: poetry run python output/fig04_cache_build.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "diagnostics"))

import spatial_cv as scv          # noqa: E402
import train_xgboost as tx        # noqa: E402
from _data import load            # noqa: E402

HERE = Path(__file__).resolve().parent
REPCV_JSON = HERE / "repeated_cv_results.json"
EXTRAP_JSON = HERE / "extrapolation_range_results.json"
OUT = HERE / "fig04_cache.npz"

OP_KM = tx.OPERATIVE_CELL_KM       # 10
N_REPEATS = 20
SEED0 = 42
LOGIT_C = 1.0
RECALL_GRID = np.linspace(0.0, 1.0, 201)


def interp_pr(y, proba, scored):
    """Interpolated precision on RECALL_GRID: precision achievable at recall >= r."""
    yv = np.asarray(y)[scored]
    pv = proba[scored]
    prec, rec, _ = precision_recall_curve(yv, pv)   # positive = class 1 (Non-abrupt)
    # For each target recall r, take the max precision among operating points with
    # recall >= r (the standard interpolated PR curve). rec/prec are paired points.
    out = np.empty_like(RECALL_GRID)
    for i, r in enumerate(RECALL_GRID):
        mask = rec >= r
        out[i] = prec[mask].max() if mask.any() else prec[-1]
    ap = average_precision_score(yv, pv)
    return out, ap


def block_median_distances(lat, lon):
    """Median great-circle distance (km) from each held-out point to nearest train
    point, pooled across folds, for each block scale (seed-42 partition)."""
    scales = tx.SWEEP_CELL_KM
    meds = []
    for s in scales:
        blocks = scv.assign_blocks(lat, lon, method=tx.BLOCK_METHOD, cell_km=s)
        folds = list(scv.buffered_block_folds(
            lat, lon, blocks, n_splits=tx.N_OUTER, buffer_km=tx.BUFFER_KM, seed=SEED0))
        d_all = []
        for tr, te in folds:
            tree = BallTree(np.radians(np.column_stack([lat[tr], lon[tr]])),
                            metric="haversine")
            d, _ = tree.query(np.radians(np.column_stack([lat[te], lon[te]])), k=1)
            d_all.append(d.ravel() * scv.EARTH_KM)
        meds.append(float(np.median(np.concatenate(d_all))))
    return list(scales), meds


def main():
    X, y, lat, lon = load(verify=True)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    prevalence = float((np.asarray(y) == 1).mean())

    hp = json.loads((tx.MODELS / "selected_hparams.json").read_text())["hyperparameters"]
    xgb_factory = tx.xgb_builder(hp)
    logit_factory = tx.logistic_builder({"C": LOGIT_C})

    # ---- Panel (a): PR curves across partitions at 10 km ---------------------
    blocks = scv.assign_blocks(lat, lon, method=tx.BLOCK_METHOD, cell_km=OP_KM)
    xgb_curves, logit_curves, xgb_aps, logit_aps = [], [], [], []
    for r in range(N_REPEATS):
        folds = list(scv.buffered_block_folds(
            lat, lon, blocks, n_splits=tx.N_OUTER, buffer_km=tx.BUFFER_KM, seed=SEED0 + r))
        xp, xs = scv.pooled_oof_predict(xgb_factory, X, y, folds)
        lp, ls = scv.pooled_oof_predict(logit_factory, X, y, folds)
        c, a = interp_pr(y, xp, xs); xgb_curves.append(c); xgb_aps.append(a)
        c, a = interp_pr(y, lp, ls); logit_curves.append(c); logit_aps.append(a)
        print(f"  repeat {r:2d}: XGB AP={xgb_aps[-1]:.3f}  logit AP={logit_aps[-1]:.3f}")

    xgb_curves = np.array(xgb_curves)
    logit_curves = np.array(logit_curves)
    xgb_aps = np.array(xgb_aps)
    logit_aps = np.array(logit_aps)

    # ---- Panel (b): block-CV median distances --------------------------------
    b_scales, b_dist = block_median_distances(lat, lon)

    rep = json.loads(REPCV_JSON.read_text())
    b_ap = [rep["per_scale"][str(int(s))]["xgb_mean"] for s in b_scales]

    ext = json.loads(EXTRAP_JSON.read_text())
    ext_rows = sorted(ext["rows"], key=lambda r: r["med_dist_km"])
    r_dist = [r["med_dist_km"] for r in ext_rows]
    r_ap = [r["ap"] for r in ext_rows]
    r_cnt = [r["regions"] for r in ext_rows]

    np.savez(
        OUT,
        prevalence=prevalence,
        recall_grid=RECALL_GRID,
        xgb_prec_mean=xgb_curves.mean(0), xgb_prec_std=xgb_curves.std(0),
        logit_prec_mean=logit_curves.mean(0), logit_prec_std=logit_curves.std(0),
        xgb_ap_mean=float(xgb_aps.mean()), xgb_ap_std=float(xgb_aps.std()),
        logit_ap_mean=float(logit_aps.mean()), logit_ap_std=float(logit_aps.std()),
        op_km=OP_KM, n_repeats=N_REPEATS,
        block_scales=np.array(b_scales, float), block_dist=np.array(b_dist, float),
        block_ap=np.array(b_ap, float),
        region_dist=np.array(r_dist, float), region_ap=np.array(r_ap, float),
        region_count=np.array(r_cnt, int),
    )
    print("\n=== cached to", OUT.name, "===")
    print(f"panel a  XGB AUC-PR {xgb_aps.mean():.3f} +/- {xgb_aps.std():.3f}  "
          f"logit {logit_aps.mean():.3f} +/- {logit_aps.std():.3f}  (floor {prevalence:.4f})")
    print("panel b  block-CV (scale km -> median dist km / AUC-PR):")
    for s, d, a in zip(b_scales, b_dist, b_ap):
        print(f"          {s:>4} km -> {d:6.1f} km / {a:.3f}")
    print("panel b  region-out (regions -> median dist km / AUC-PR):")
    for c, d, a in zip(r_cnt, r_dist, r_ap):
        print(f"          {c:>4}    -> {d:6.1f} km / {a:.3f}")


if __name__ == "__main__":
    main()
