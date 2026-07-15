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

Positive class = 1 (Non-abrupt, minority); headline metric AUC-PR (chance = prevalence).
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
          f"pos(Non-abrupt) prevalence={prev:.4f}  AUC-PR floor={prev:.4f}\n")

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
    # CRITICAL: search only the *informative* regime. Once the buffer strips so
    # much train data that the model can no longer discriminate, AUC-PR collapses
    # to the prevalence floor and ROC to 0.5 -- a long flat *dead* tail that a
    # naive flatness test would mis-read as a plateau. Require the window to stay
    # above the chance floor and keep real ROC signal, so a collapsed floor is
    # never reported as the recommended buffer.
    aps = np.array([row[3] for row in results], dtype=float)
    rocs = np.array([row[6] for row in results], dtype=float)
    informative = (rocs > 0.55) & (aps > prev + 0.02)
    plateau_r = None
    for i in range(len(aps) - 3):
        window = aps[i:i + 4]
        if (np.all(np.isfinite(window)) and informative[i:i + 4].all()
                and (window.max() - window.min()) < 0.01):
            plateau_r = results[i][0]
            break
    print("\n" + "=" * 62)
    r0_ap = results[0][3]
    pl_ap = None
    if plateau_r is not None:
        pl_ap = aps[[row[0] for row in results].index(plateau_r)]
        print(f"random-split (r=0) AUC-PR = {r0_ap:.4f}  (leaky reference)")
        print(f"targeted curve plateaus at r ~= {plateau_r} km, AUC-PR ~= {pl_ap:.4f}")
        print(f"=> recommended interpolation-case (A) buffer ~= {plateau_r} km")
    else:
        print("no clear plateau within 40 km (training pool likely depleted first) --")
        print("read the table: the informative regime is where 'removed%' < ~80%.")
    print("At large r 'removed%' -> ~100%: a dispersed random test set cannot sustain")
    print("a large clean buffer. That is the argument for BLOCK holdout in case (B).")

    # Leakage-specific range: the initial CONTIGUOUS run of radii (from the
    # smallest buffer up) where targeted removal costs more AUC-PR than matched-
    # count random removal. Only the contiguous run is trustworthy -- once the
    # pool is depleted the gap sign flips randomly on tiny unstable fits, so a
    # "last positive gap" reading would report late noise, not real leakage.
    GAP_EPS = 0.02
    leak_r = None
    for row in results:
        r_km, keep_n, nrm, ap, ctl_ap, gap, roc = row
        if r_km == 0:
            continue  # gap is 0 by construction at r=0 (identical train sets)
        if np.isfinite(gap) and gap > GAP_EPS:
            leak_r = r_km
        else:
            break     # first radius the gap drops into noise ends the run
    if leak_r is not None:
        print(f"leakage-specific signal (targeted-vs-control gap > {GAP_EPS}) is present "
              f"and contiguous through r ~= {leak_r} km; beyond it the gap collapses into "
              f"data-depletion noise (pool already >90% stripped).")
    else:
        print("no contiguous targeted-vs-control gap above noise -- leakage not "
              "separable from data loss in this design.")

    # Buffer-sensitivity readout at the task's canonical radii (T43).
    by_r = {row[0]: row for row in results}
    print("\nbuffer-sensitivity sweep (targeted AUC-PR at canonical radii):")
    for rq in (1, 2, 5, 10):
        if rq in by_r:
            _, _, nrm, ap, _, _, _ = by_r[rq]
            print(f"  {rq:>2} km: AUC-PR={ap:.4f}  ({100*nrm/n_tr:.1f}% train removed)")

    try:
        _plot(results, prev, plateau_r, r0_ap, pl_ap)
    except Exception as e:
        print(f"(plot skipped: {e})")


def _plot(results, prev, plateau_r=None, r0_ap=None, pl_ap=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = np.array([x[0] for x in results], dtype=float)
    tgt = np.array([x[3] for x in results], dtype=float)
    ctl = np.array([x[4] for x in results], dtype=float)
    roc = np.array([x[6] for x in results], dtype=float)
    removed = [100 * x[2] / (x[1] + x[2]) if (x[1] + x[2]) else 0 for x in results]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Grey out the depletion dead-zone: once the buffer strips so much train data
    # that the model no longer discriminates (ROC collapses to ~0.5), both curves
    # are meaningless and their gap is depletion noise, NOT leakage. Mark it so the
    # eye doesn't read that (large) region as signal.
    informative = (roc > 0.55) & np.isfinite(tgt) & np.isfinite(ctl) & (tgt > prev + 0.02)
    dep = np.where(~informative)[0]
    dep_start = r[dep[0]] if len(dep) and dep[0] > 0 else None
    if dep_start is not None:
        ax.axvspan(dep_start, r[-1], color='0.85', alpha=0.5, zorder=0)
        ax.text(dep_start + 0.4, prev + 0.03, 'training pool depleted\n(fits degenerate -> chance)',
                fontsize=8, color='0.35', va='bottom')

    # Shade the targeted<->control gap ONLY in the informative regime: there the
    # band is real leakage (targeted drops below control because it strips
    # spatially-leaky rows). Outside it the gap is not leakage, so it is not shaded.
    ax.fill_between(r, tgt, ctl, where=informative, color='tab:orange', alpha=0.20,
                    label='leakage (targeted vs. control gap)')

    ax.plot(r, tgt, 'o-', lw=2, ms=4, color='tab:blue',
            label='targeted removal (buffer) = honest')
    ax.plot(r, ctl, 's--', lw=2, ms=4, color='gray',
            label='random removal (matched n) = control')
    ax.axhline(prev, color='k', ls=':', lw=1, label=f'AUC-PR chance floor ({prev:.3f})')

    # Make the drop explicit: annotate the leaky r=0 start and the plateau.
    if r0_ap is not None and np.isfinite(r0_ap):
        ax.annotate(f'r=0 (leaky): {r0_ap:.3f}', xy=(0, r0_ap),
                    xytext=(4, r0_ap + 0.04), fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1))
    if plateau_r is not None:
        ax.axvline(plateau_r, color='tab:green', ls='-', lw=1.4, alpha=0.8)
        if pl_ap is not None and np.isfinite(pl_ap):
            ax.annotate(f'plateau ~{plateau_r} km: {pl_ap:.3f}\n(recommended buffer)',
                        xy=(plateau_r, pl_ap), xytext=(plateau_r + 3, pl_ap + 0.10),
                        fontsize=9, color='tab:green',
                        arrowprops=dict(arrowstyle='->', color='tab:green', lw=1))
        if r0_ap is not None and pl_ap is not None and np.isfinite(r0_ap) and np.isfinite(pl_ap):
            ax.set_title(f'Leakage-decay curve: AUC-PR falls {r0_ap:.3f} -> {pl_ap:.3f} '
                         f'({r0_ap - pl_ap:+.3f}) as train/test separation grows')
    if ax.get_title() == '':
        ax.set_title('Leakage-decay curve: AUC-PR vs enforced train/test separation')

    ax.set_xlabel('spatial exclusion buffer between train and test (km)')
    ax.set_ylabel('AUC-PR (positive = Non-abrupt)')
    ax.legend(loc='center right', fontsize=8)
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
