"""Leakage-decay curve: size the spatial buffer empirically.

Settled with the user (develop-model interview, 2026-07-08):
  * Start from the SAME random split verify-ml used (seed 8) so buffer r=0
    reproduces the leaky AUC-PR ~= 0.90 reported by baseline_and_shuffle.py.
  * Sweep a spatial exclusion buffer r = 0..40 km in 1 km steps: at each r,
    drop every TRAIN point within r (great-circle) of ANY test point, refit,
    score the fixed test set. AUC-PR falls and plateaus; the plateau is the
    effective leakage range and sets the interpolation-case (A) buffer.
  * CONTROL curve: remove the SAME NUMBER of train points AT RANDOM (matched
    sample size, spatially untargeted). The gap targeted-vs-control separates
    leakage (what we want to remove) from mere data loss (a confound). The
    honest buffer is where the TARGETED curve plateaus, cross-checked that the
    gap is what's driving the decline there.

Positive class = 1 (Gradual, minority); headline metric AUC-PR (chance = prevalence).
Distances are great-circle via a haversine BallTree (lat/lon in Alaska: planar
degrees would distort ~2x between 57N and 71N, so we do NOT use Euclidean degrees).

Run: poetry run python diagnostics/leakage_decay.py
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.metrics import average_precision_score, roc_auc_score

from _data import load

EARTH_KM = 6371.0088
SEED = 42
RADII_KM = np.arange(0, 41, 1)   # 0..40 km, 1 km steps


def outside_buffer_mask(train_lat, train_lon, test_lat, test_lon, r_km):
    """Return bool mask over TRAIN points: True == keep (no test point within r_km).

    Great-circle distance via a haversine BallTree built on the test points.
    r_km == 0 keeps everything (a point is never strictly *within* a 0 radius of
    a distinct neighbour; coincident points are handled as within-buffer).
    """
    if r_km <= 0:
        return np.ones(len(train_lat), dtype=bool)
    test_rad = np.radians(np.column_stack([test_lat, test_lon]))
    train_rad = np.radians(np.column_stack([train_lat, train_lon]))
    tree = BallTree(test_rad, metric='haversine')
    n_within = tree.query_radius(train_rad, r=r_km / EARTH_KM, count_only=True)
    return n_within == 0


def _xgb(spw):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, min_child_weight=20, learning_rate=0.05,
        reg_lambda=50, gamma=1, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
        scale_pos_weight=spw, random_state=SEED)


def _fit_score(Xtr, ytr, Xte, yte):
    """Fit on a training subset, return (AUC-PR, AUC-ROC) on the fixed test set.

    Returns (nan, nan) if the subset lacks both classes or has no positives.
    """
    if len(np.unique(ytr)) < 2 or (ytr == 1).sum() < 1:
        return np.nan, np.nan
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = _xgb(spw).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return average_precision_score(yte, p), roc_auc_score(yte, p)


def _selftest():
    """Cheap self-checks on the buffer primitive before the sweep runs."""
    # r=0 keeps everything.
    assert outside_buffer_mask(np.array([60.]), np.array([-150.]),
                               np.array([60.]), np.array([-150.]), 0).all()
    # A train point ~1.1 km north of a test point: removed at 2 km, kept at 0.5 km.
    tlat, tlon = np.array([65.01]), np.array([-150.0])   # ~1.11 km north
    xlat, xlon = np.array([65.00]), np.array([-150.0])
    assert outside_buffer_mask(tlat, tlon, xlat, xlon, 2.0)[0] == False, "should be removed at 2km"
    assert outside_buffer_mask(tlat, tlon, xlat, xlon, 0.5)[0] == True, "should be kept at 0.5km"
    # A distant train point (~111 km) is kept at 40 km.
    assert outside_buffer_mask(np.array([66.0]), np.array([-150.0]),
                               xlat, xlon, 40.0)[0] == True
    print("[selftest] buffer primitive OK")


def main():
    _selftest()
    X, y, lat, lon = load(verify=True)
    yv = y.to_numpy()
    prev = yv.mean()

    rng = np.random.default_rng(SEED)
    split_seed = int(rng.integers(0, 100))  # == 8, matches verify-ml
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=0.3, random_state=split_seed,
                                shuffle=True, stratify=y)
    Xtr_all, ytr_all = X.iloc[itr], yv[itr]
    Xte, yte = X.iloc[ite], yv[ite]
    lat_tr, lon_tr, lat_te, lon_te = lat[itr], lon[itr], lat[ite], lon[ite]
    n_tr = len(itr)
    print(f"split seed={split_seed}  train={n_tr}  test={len(ite)}  "
          f"pos(Gradual) prevalence={prev:.4f}  AUC-PR floor={prev:.4f}\n")

    print(f"{'r_km':>5} {'keep':>6} {'removed%':>9} {'tgt_AUCPR':>10} "
          f"{'ctl_AUCPR':>10} {'gap':>7} {'tgt_ROC':>8}")
    print("-" * 62)
    results = []
    ctl_rng = np.random.default_rng(SEED)
    for r in RADII_KM:
        keep = outside_buffer_mask(lat_tr, lon_tr, lat_te, lon_te, float(r))
        n_removed = int((~keep).sum())
        tgt_ap, tgt_roc = _fit_score(Xtr_all[keep], ytr_all[keep], Xte, yte)

        # Control: drop the same COUNT at random from the full training pool.
        if n_removed == 0:
            ctl_ap = tgt_ap
        else:
            perm = ctl_rng.permutation(n_tr)
            ctl_keep = np.ones(n_tr, dtype=bool)
            ctl_keep[perm[:n_removed]] = False
            ctl_ap, _ = _fit_score(Xtr_all[ctl_keep], ytr_all[ctl_keep], Xte, yte)

        gap = (ctl_ap - tgt_ap) if np.isfinite(ctl_ap) and np.isfinite(tgt_ap) else np.nan
        results.append((int(r), int(keep.sum()), n_removed, tgt_ap, ctl_ap, gap, tgt_roc))
        print(f"{r:>5} {keep.sum():>6} {100*n_removed/n_tr:>8.1f}% {tgt_ap:>10.4f} "
              f"{ctl_ap:>10.4f} {gap:>7.4f} {tgt_roc:>8.4f}")

    # Plateau of the TARGETED curve: first r where the next 3 km change < 0.01.
    aps = np.array([row[3] for row in results], dtype=float)
    plateau_r = None
    for i in range(len(aps) - 3):
        window = aps[i:i + 4]
        if np.all(np.isfinite(window)) and (window.max() - window.min()) < 0.01:
            plateau_r = results[i][0]
            break
    print("\n" + "=" * 62)
    if plateau_r is not None:
        r0_ap = results[0][3]
        pl_ap = aps[[row[0] for row in results].index(plateau_r)]
        print(f"random-split (r=0) AUC-PR = {r0_ap:.4f}  (leaky reference)")
        print(f"targeted curve plateaus at r ~= {plateau_r} km, AUC-PR ~= {pl_ap:.4f}")
        print(f"=> recommended interpolation-case (A) buffer ~= {plateau_r} km")
    else:
        print("no clear plateau within 40 km (training pool likely depleted first) --")
        print("read the table: the informative regime is where 'removed%' < ~80%.")
    print("At large r 'removed%' -> ~100%: a dispersed random test set cannot sustain")
    print("a large clean buffer. That is the argument for BLOCK holdout in case (B).")

    try:
        _plot(results, prev)
    except Exception as e:
        print(f"(plot skipped: {e})")


def _plot(results, prev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = [x[0] for x in results]
    tgt = [x[3] for x in results]
    ctl = [x[4] for x in results]
    removed = [100 * x[2] / (x[1] + x[2]) if (x[1] + x[2]) else 0 for x in results]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(r, tgt, 'o-', lw=2, ms=4, label='targeted removal (buffer) = honest')
    ax.plot(r, ctl, 's--', lw=2, ms=4, color='gray', label='random removal (matched n) = control')
    ax.axhline(prev, color='k', ls=':', lw=1, label=f'AUC-PR chance floor ({prev:.3f})')
    ax.set_xlabel('spatial exclusion buffer between train and test (km)')
    ax.set_ylabel('AUC-PR (positive = Gradual)')
    ax.set_title('Leakage-decay curve: AUC-PR vs enforced train/test separation')
    ax.legend(loc='center right')
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(r, removed, color='tab:red', alpha=0.35, lw=1)
    ax2.set_ylabel('% of training points removed', color='tab:red')
    ax2.set_ylim(0, 100)
    fig.tight_layout()
    out = 'diagnostics/leakage_decay.png'
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == '__main__':
    main()
