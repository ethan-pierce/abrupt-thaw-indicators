"""Buffered spatial-block cross-validation for the thaw-mode classifier.

The single protocol settled with the user (develop-model, 2026-07-08), replacing
the leaky random split. Points are grouped into contiguous spatial BLOCKS; whole
blocks are held out together; an optional dead-zone BUFFER removes training points
within `buffer_km` (great-circle) of the held-out block, killing cross-boundary
leakage. Block SIZE selects the inference regime:
  * small blocks  -> interpolation (case A): serves the statewide map (Headline A)
  * large blocks  -> extrapolation  (case B): stress-tests the SHAP story (Headline C)

Estimator-agnostic by design: this module only splits and pools predictions, so it
can be exercised and checked in isolation (verify-ml / verify-code). The model
factory lives in the caller. Distances are great-circle (haversine); planar
degrees would distort ~2x across Alaska's 57N-71N span, so they are never used.

References for the method: Roberts et al. 2017 (Ecography); Valavi et al. 2019 (blockCV).
"""
import numpy as np
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088


# --------------------------------------------------------------------------
# geography
# --------------------------------------------------------------------------
def _equirect_xy(lat, lon):
    """Aspect-corrected planar coords (km-ish) for compact clustering only.

    x scaled by cos(mean lat) so a degree of lon and lat are comparable; used for
    grouping geometry, NOT for the buffer (the buffer uses true great-circle).
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    x = lon * np.cos(np.radians(lat.mean()))
    return np.column_stack([x, lat])


def within_radius_of_set(query_lat, query_lon, ref_lat, ref_lon, r_km):
    """Bool mask over QUERY points: True == within r_km great-circle of any ref point."""
    if r_km <= 0 or len(ref_lat) == 0:
        return np.zeros(len(query_lat), dtype=bool)
    ref = np.radians(np.column_stack([np.asarray(ref_lat, float), np.asarray(ref_lon, float)]))
    q = np.radians(np.column_stack([np.asarray(query_lat, float), np.asarray(query_lon, float)]))
    tree = BallTree(ref, metric='haversine')
    n = tree.query_radius(q, r=r_km / EARTH_KM, count_only=True)
    return n > 0


# --------------------------------------------------------------------------
# block assignment
# --------------------------------------------------------------------------
ALBERS_EPSG = 3338  # NAD83 / Alaska Albers, equal-area metres


def _albers_xy(lat, lon):
    """Project lat/lon (EPSG:4326) to Alaska Albers (EPSG:3338) metres.

    Equal-area, so a fixed cell edge in metres is a true, interpretable ground
    distance everywhere across Alaska's span (unlike degree cells, which shrink
    ~2x in longitude from 57N to 71N).
    """
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, ALBERS_EPSG, always_xy=True)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x, float), np.asarray(y, float)


def assign_blocks(lat, lon, method='grid', cell_deg=1.0, cell_km=10.0,
                  n_clusters=25, seed=0):
    """Return an integer block id per point.

    method='grid'        : lat/lon binned to `cell_deg` cells (simple, reproducible).
    method='albers_grid' : Alaska Albers (EPSG:3338) equal-area grid; points binned
                           to square cells of `cell_km` edge length in metres — a
                           real, interpretable ground scale for the block-size sweep.
    method='kmeans'      : `n_clusters` compact contiguous clusters on aspect-corrected
                           coords (roughly balanced counts, no empty cells).
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if method == 'grid':
        return (np.floor(lat / cell_deg).astype(int) * 100000
                + np.floor(lon / cell_deg).astype(int))
    if method == 'albers_grid':
        x, y = _albers_xy(lat, lon)
        edge_m = cell_km * 1000.0
        ix = np.floor(x / edge_m).astype(np.int64)
        iy = np.floor(y / edge_m).astype(np.int64)
        # Bijective pairing over the (bounded) Alaska cell-index range; the offset
        # keeps indices non-negative so the id is stable and collision-free.
        return (ix + 1_000_000) * 10_000_000 + (iy + 1_000_000)
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        xy = _equirect_xy(lat, lon)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(xy)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------
def buffered_block_folds(lat, lon, blocks, n_splits=5, buffer_km=0.0, seed=0):
    """Yield (train_idx, test_idx) for buffered block CV.

    Whole blocks are partitioned into `n_splits` folds (shuffled by block, so a
    block is never split across train and test). For each fold, test = points in
    the held-out blocks; train = points in the remaining blocks MINUS any within
    `buffer_km` great-circle of a test point (the dead zone). Every point is a test
    point exactly once -> supports pooled out-of-fold scoring.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    block_fold = {b: i % n_splits for i, b in enumerate(uniq)}
    fold_of_point = np.array([block_fold[b] for b in blocks])
    idx = np.arange(len(blocks))

    for f in range(n_splits):
        test_idx = idx[fold_of_point == f]
        cand_train = idx[fold_of_point != f]
        if buffer_km > 0 and len(test_idx):
            in_zone = within_radius_of_set(lat[cand_train], lon[cand_train],
                                           lat[test_idx], lon[test_idx], buffer_km)
            cand_train = cand_train[~in_zone]
        yield cand_train, test_idx


def nested_block_folds(lat, lon, blocks, n_splits_outer=5, n_splits_inner=5,
                       buffer_km=0.0, seed=0):
    """Yield nested buffered block folds for double-dip-free selection (B4).

    For each OUTER fold produced by `buffered_block_folds` over ALL points, run
    `buffered_block_folds` again over the outer-TRAIN subset (which already has the
    buffer dead-zone removed) to produce INNER folds for hyperparameter selection.

    Yields `(outer_train_idx, outer_test_idx, inner_folds)` where `inner_folds` is a
    materialized list of `(inner_train_idx, inner_val_idx)` tuples. All indices refer
    to the ORIGINAL arrays. Because inner folds are drawn only from the outer-train
    subset, no inner point ever falls in an outer-test block.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    blocks = np.asarray(blocks)
    outer = buffered_block_folds(lat, lon, blocks, n_splits=n_splits_outer,
                                 buffer_km=buffer_km, seed=seed)
    for f, (outer_train_idx, outer_test_idx) in enumerate(outer):
        # Inner CV runs on the outer-train subset only; a distinct per-fold seed
        # keeps the inner block shuffle deterministic yet decorrelated from outer.
        inner = buffered_block_folds(lat[outer_train_idx], lon[outer_train_idx],
                                     blocks[outer_train_idx], n_splits=n_splits_inner,
                                     buffer_km=buffer_km, seed=seed + f + 1)
        inner_folds = [(outer_train_idx[itr], outer_train_idx[iva])
                       for itr, iva in inner]
        yield outer_train_idx, outer_test_idx, inner_folds


def fold_class_counts(folds, y, positive=1):
    """Per-fold train/test sizes and minority (positive=Non-abrupt, class 1) counts.

    Takes a materialized list of `(train_idx, test_idx)` folds (flat or the inner
    folds from `nested_block_folds`) and returns a list of dicts. Lets the caller
    log Non-abrupt sparsity per fold at each block size (B5b): at large blocks some
    folds may hold few or zero Non-abrupt points, which pooled scoring must survive.
    """
    y = np.asarray(y)
    rows = []
    for f, (train_idx, test_idx) in enumerate(folds):
        rows.append({
            'fold': f,
            'train_n': int(len(train_idx)),
            'train_pos': int((y[train_idx] == positive).sum()),
            'test_n': int(len(test_idx)),
            'test_pos': int((y[test_idx] == positive).sum()),
        })
    return rows


def pooled_oof_predict(estimator_factory, X, y, folds):
    """Fit per fold, collect out-of-fold P(class=1); return (proba, mask_scored).

    `estimator_factory(y_train)` returns a fresh estimator (lets the caller set
    scale_pos_weight from the fold's own class balance). Test points in a fold
    whose training subset lacks a positive class are left unscored (mask False).
    Pool AUC-PR/ROC on the scored points -> robust to folds sparse in the minority.
    """
    y = np.asarray(y)
    proba = np.full(len(y), np.nan)
    scored = np.zeros(len(y), dtype=bool)
    for train_idx, test_idx in folds:
        ytr = y[train_idx]
        if len(np.unique(ytr)) < 2 or (ytr == 1).sum() < 1 or len(test_idx) == 0:
            continue
        est = estimator_factory(ytr)
        est.fit(X.iloc[train_idx], ytr)
        proba[test_idx] = est.predict_proba(X.iloc[test_idx])[:, 1]
        scored[test_idx] = True
    return proba, scored


# --------------------------------------------------------------------------
# self-checks
# --------------------------------------------------------------------------
def _selftest():
    # within_radius_of_set: ~1.11 km north point is inside a 2 km ref buffer, outside 0.5 km.
    assert within_radius_of_set([65.01], [-150.], [65.00], [-150.], 2.0)[0]
    assert not within_radius_of_set([65.01], [-150.], [65.00], [-150.], 0.5)[0]

    # albers_grid: two far-apart points land in different cells; two points closer
    # than the cell edge share a cell; ids are stable/reproducible across calls.
    la = np.array([60.00, 60.02, 70.00])   # first two ~2.2 km apart; third ~1100 km north
    lo = np.array([-150., -150., -140.])
    big = assign_blocks(la, lo, method='albers_grid', cell_km=50.0)
    assert big[0] == big[1] and big[0] != big[2]          # 2.2 km < 50 km cell; 1100 km apart differ
    assert np.array_equal(big, assign_blocks(la, lo, method='albers_grid', cell_km=50.0))  # stable
    small = assign_blocks(la, lo, method='albers_grid', cell_km=1.0)
    assert small[0] != small[1]                            # 2.2 km > 1 km cell -> separate

    # Synthetic: two far-apart clusters -> kmeans(2) separates them; block CV holds
    # a whole cluster out; buffer removes near-boundary train points.
    lat = np.array([60., 60.01, 60.02, 70., 70.01, 70.02])
    lon = np.array([-150., -150., -150., -140., -140., -140.])
    blocks = assign_blocks(lat, lon, method='kmeans', n_clusters=2, seed=0)
    assert len(np.unique(blocks)) == 2
    assert blocks[0] == blocks[1] == blocks[2] and blocks[3] == blocks[4] == blocks[5]

    # Each point is a test point exactly once across folds.
    seen = np.zeros(6, dtype=int)
    for tr, te in buffered_block_folds(lat, lon, blocks, n_splits=2, buffer_km=0.0, seed=0):
        seen[te] += 1
        assert set(tr).isdisjoint(set(te))
    assert (seen == 1).all()

    # Buffer removes near neighbours of the held-out block from train.
    lat2 = np.array([60.0, 60.005, 70.0])   # first two ~0.55 km apart, in block A
    lon2 = np.array([-150., -150., -140.])
    b2 = np.array([0, 0, 1])                 # block 0 held out -> train candidates are idx 2 only anyway
    # hold out block 1 (idx2); buffer shouldn't remove the distant block-0 points
    folds = list(buffered_block_folds(lat2, lon2, b2, n_splits=2, buffer_km=1.0, seed=0))

    # Nested folds: inner train/val never touch an outer-test block. Build a few
    # blocks so both outer and inner splits are non-degenerate.
    latn = np.array([60., 60.01, 62., 62.01, 68., 68.01, 70., 70.01])
    lonn = np.array([-150., -150., -148., -148., -142., -142., -140., -140.])
    bn = assign_blocks(latn, lonn, method='kmeans', n_clusters=4, seed=0)
    for otr, ote, inner in nested_block_folds(latn, lonn, bn, n_splits_outer=2,
                                              n_splits_inner=2, buffer_km=0.0, seed=0):
        test_blocks = set(bn[ote])
        for itr, iva in inner:
            assert set(itr).isdisjoint(set(ote)) and set(iva).isdisjoint(set(ote))
            assert set(bn[itr]).isdisjoint(test_blocks)
            assert set(bn[iva]).isdisjoint(test_blocks)

    # fold_class_counts: totals across folds match the data; each Non-abrupt point is
    # a test-positive in exactly one fold (pooled-OOF property).
    yc = np.array([0, 0, 1, 0, 1, 0, 0, 1])   # 3 Non-abrupt (class 1) among 8 points
    cfolds = list(buffered_block_folds(latn, lonn, bn, n_splits=2, buffer_km=0.0, seed=0))
    counts = fold_class_counts(cfolds, yc, positive=1)
    assert sum(r['test_n'] for r in counts) == len(yc)
    assert sum(r['test_pos'] for r in counts) == int((yc == 1).sum())

    print("[spatial_cv selftest] OK")


if __name__ == '__main__':
    _selftest()
