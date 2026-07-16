"""Area-of-Applicability (AOA) reliability layer for the statewide thaw map [T21/G18].

Method: Meyer & Pebesma 2021 (Methods Ecol. Evol.), "Predicting into unknown space?
Estimating the area of applicability of spatial prediction models". A late-paper
*caveat* layer, NOT a headline and NOT folded into the susceptibility surface -- the
Obu permafrost domain (T20) answers "is abrupt-vs-non-abrupt even defined here?"; the
AOA answers the separate question "does this grid cell fall inside the feature-space the
model actually learned from, or is the score an extrapolation?" (T20 design note:
"Reliability stays a separate layer (T21/AOA).")

WHY NOT LITERAL M&P (raw standardized-z distance):
  The literal metric is Euclidean over train-standardized z. A handful of features have
  a grid spread 50-77x the training spread (genuine values: Yukon-scale drainage,
  icefield summer temps, glacier SWE trends -- not fill bugs), so one band's heavy tail
  dictates the distance: the "dissimilarity" becomes univariate-in-disguise (a single
  feature contributes up to 93% of it) and the outside-AOA fraction swings 42-64% with
  the arbitrary transform choice. Per-feature transforms just move the domination to the
  next feature. We fix the COORDINATE instead of the transform (T21 handoff):

RANK -> TRAINING-CDF COORDINATE (the operative metric):
  Each CONTINUOUS predictor v is mapped to its empirical training CDF rank
  F_train(v) in [0, 1] (np.interp on the sorted training values). Beyond the training
  min/max the map is LINEARLY EXTENDED in robust (IQR) units -- 1 + (v-max)/IQR above,
  (v-min)/IQR below -- so genuinely out-of-range cells still register as extrapolation
  instead of clipping to 1.0. Rationale: XGBoost splits are rank-based (monotone-
  invariant), so rank space is the coordinate the model actually perceives; it is
  bounded, so no single feature can dominate (max single-feature share falls 93% -> ~12%
  empirically) and it needs no per-feature transform tuning. This DEPARTS from literal
  M&P; the DI-vs-CV-performance calibration (diagnostics/aoa_calibration.py) is its
  justification -- rank-CDF DI predicts OOF skill degradation, which is the property an
  applicability index must have. BINARY one-hots (26: Land Cover / Vegetation Mode /
  Yedoma) are NOT rank-mapped -- they pass through as 0/1 (a rank CDF of a two-value
  column is meaningless). Every coordinate is then weighted by mean|SHAP| (below).

Algorithm (importance-weighted dissimilarity index, DI):
  1. Map each predictor to the rank-CDF coordinate above (binaries stay 0/1).
  2. Weight each coordinate by its variable importance = mean|SHAP| from the operative
     all-data model (the project's canonical importance currency; T25/T41), normalized to
     sum 1 -- so features the model barely uses do not drive the dissimilarity geometry.
  3. dbar = mean pairwise Euclidean distance among training points in weighted rank-CDF
     space (the natural dissimilarity scale of the training set). NOTE: dbar cancels out
     of the in/out flag (both the grid DI and the CV-DI threshold divide by it) -- it only
     sets the DI's absolute scale, not the classification.
  4. DI(cell) = (distance from the cell to its NEAREST training point) / dbar.
  5. Threshold: anchored to CV performance where possible (diagnostics/aoa_calibration.py
     writes models/aoa_threshold.json = the DI at which pooled-OOF AUC-PR for Non-abrupt
     falls toward the prevalence floor). If that file is absent we fall back to the
     box-plot outlier fence Q75 + 1.5*IQR of the CV training-DI distribution (same-fold
     neighbours excluded), and say so in the provenance.
  6. A cell is INSIDE the AOA (reliable) iff DI(cell) <= threshold, else it is flagged as
     extrapolating beyond the training feature distribution.

NaN handling: reuse predict.py's Obu mask exactly (PerProb>0 AND >=1 feature); NaN
off-domain. A missing CONTINUOUS predictor is imputed to rank 0.5 (the training median in
rank space = "a typical value", so it contributes nothing extreme to the distance -- the
faithful translation of the settled "impute to the feature mean" rule into rank space,
where the mean/centre is 0.5, not 0). A missing BINARY one-hot -> 0 (absent category, the
same convention the model's own preprocessing uses). Applied identically to training
points and grid cells, so dbar, the threshold, and the grid DI share one convention.

Output (a reliability raster, aligned to susceptibility.nc and NaN off the Obu domain):
  data/aoa.nc              -- DI (continuous, primary) + inside_aoa (derived binary flag)
  output/aoa_map.png       -- binary applicability map
  output/aoa_di_map.png    -- continuous dissimilarity-index map
  + a per-feature "drivers of extrapolation" readout to stdout (which features carry the
    distance of the outside-AOA cells).

Run: poetry run python models/aoa.py       (AOA_SMOKE=1 subsamples the grid for a fast check)

This module is import-safe: the heavy load/score pipeline lives under main(); the
coordinate + distance helpers are importable (diagnostics/aoa_calibration.py reuses them
to calibrate the threshold on the identical metric).
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block
import matplotlib.pyplot as plt
import xarray as xr
import xgboost as xgb
import shap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT
from data import local_rasters
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_cv import assign_blocks, buffered_block_folds

# Operative CV protocol -- MUST match train_xgboost / cv_config.json so the threshold
# folds are the same spatial-block scheme the model was selected under.
BLOCK_METHOD = 'albers_grid'
OPERATIVE_CELL_KM = 10
BUFFER_KM = 0.0
N_SPLITS = 5
CV_SEED = 42

THRESHOLD_JSON = MODELS / 'aoa_threshold.json'  # written by diagnostics/aoa_calibration.py

SMOKE = bool(os.environ.get('AOA_SMOKE'))
CHUNK = 20000  # grid rows per nearest-distance chunk (bounds the BLAS distance matrix)


# ======================================================================================
# Distance helpers (chunked BLAS) -- importable, estimator-free.
# ======================================================================================
def nearest_distance(query, ref, ref_norm=None, chunk=CHUNK, return_index=False):
    """Min Euclidean distance from each `query` row to any `ref` row (chunked BLAS).

    d^2 = ||q||^2 + ||r||^2 - 2 q.r  -- a matmul per chunk, so 2.8M x 19k stays fast
    without ever materializing the full distance matrix. With `return_index=True` also
    returns the index of the nearest `ref` row per query (for the extrapolation-drivers
    decomposition).
    """
    ref = np.ascontiguousarray(ref, dtype=np.float64)
    if ref_norm is None:
        ref_norm = (ref ** 2).sum(axis=1)
    out = np.empty(len(query), dtype=np.float64)
    idx = np.empty(len(query), dtype=np.int64) if return_index else None
    for s in range(0, len(query), chunk):
        q = np.ascontiguousarray(query[s:s + chunk], dtype=np.float64)
        qnorm = (q ** 2).sum(axis=1)
        # NumPy's SIMD matmul raises spurious divide/overflow/invalid FPE flags from
        # masked vector lanes even on finite inputs; the result is correct to machine
        # precision (verified vs einsum), so silence the flags around it only.
        with np.errstate(all='ignore'):
            d2 = qnorm[:, None] + ref_norm[None, :] - 2.0 * (q @ ref.T)
        np.maximum(d2, 0.0, out=d2)  # clip tiny negatives from round-off
        j = d2.argmin(axis=1)
        out[s:s + chunk] = np.sqrt(d2[np.arange(len(q)), j])
        if return_index:
            idx[s:s + chunk] = j
    return (out, idx) if return_index else out


def mean_pairwise_distance(pts, chunk=2000):
    """Exact mean pairwise Euclidean distance among `pts` (chunked upper triangle)."""
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    n = len(pts)
    pnorm = (pts ** 2).sum(axis=1)
    total, count = 0.0, 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = pts[s:e]
        with np.errstate(all='ignore'):  # spurious SIMD-matmul FPE flags (see nearest_distance)
            d2 = pnorm[s:e, None] + pnorm[None, :] - 2.0 * (block @ pts.T)
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2)
        # sum only pairs (i<j): for rows [s:e], count columns strictly greater than the row.
        for i in range(e - s):
            gi = s + i
            total += d[i, gi + 1:].sum()
            count += n - gi - 1
    return total / count


# ======================================================================================
# Coordinate construction -- importable so the calibration diagnostic uses the identical
# metric it is meant to justify.
# ======================================================================================
def detect_binary(X):
    """Bool mask over columns of a DataFrame whose non-NaN values are all in {0, 1}.

    Detected by VALUE (not name) so it survives renames -- the 26 one-hot Land Cover /
    Vegetation Mode indicators and Yedoma. Identical rule to train_xgboost._binary_cols.
    """
    mask = np.zeros(X.shape[1], dtype=bool)
    for j, c in enumerate(X.columns):
        u = pd.unique(X[c].dropna())
        if len(u) and set(np.asarray(u, dtype=float).tolist()) <= {0.0, 1.0}:
            mask[j] = True
    return mask


def fit_rank_cdf(X_train, binary_mask):
    """Fit the per-feature rank->training-CDF map on the training matrix.

    Returns a list (one entry per feature). For a CONTINUOUS feature the entry is a dict
    with the sorted non-NaN training values, their CDF positions (linspace 0..1), and the
    training min/max/IQR used for the linear out-of-range extension. Binary features get
    None (they pass through as 0/1 -- no rank map).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    fitted = []
    for j in range(X_train.shape[1]):
        if binary_mask[j]:
            fitted.append(None)
            continue
        col = X_train[:, j]
        xs = np.sort(col[np.isfinite(col)])
        if len(xs) == 0:
            fitted.append(None)  # all-NaN continuous column -> pass through (imputed to 0.5)
            continue
        ps = np.linspace(0.0, 1.0, len(xs))
        q25, q75 = np.percentile(xs, [25, 75])
        iqr = q75 - q25
        if iqr <= 0:  # degenerate spread -> fall back to full range so extension is finite
            iqr = (xs[-1] - xs[0]) or 1.0
        fitted.append({'xs': xs, 'ps': ps, 'tmin': float(xs[0]),
                       'tmax': float(xs[-1]), 'iqr': float(iqr)})
    return fitted


# Out-of-range extension is capped at this many coordinate units beyond [0,1]. The
# extension is graded in IQR units within OOR_CAP of the training extreme, then SATURATES.
# The cap is essential: a bare linear IQR extension is UNBOUNDED for heavy-tailed features
# (Upstream Area's grid values reach ~40x its training max in ~1e6 tiny-IQR units), which
# re-creates exactly the single-feature domination rank space exists to remove (one
# feature -> 100% of the distance). Capping keeps genuinely out-of-range cells FLAGGED as
# more-extreme-than-any-training-value (coord 1+cap vs in-range max 1) yet BOUNDED, so no
# feature can dominate -- the plan's "keeps out-of-range flagged but bounded".
OOR_CAP = 1.0  # coord range [-OOR_CAP, 1+OOR_CAP]


def transform_rank_cdf(arr, fitted, binary_mask, oor_cap=OOR_CAP):
    """Map a raw (rows, n_features) matrix to rank-CDF coordinates. Continuous: F_train(v)
    in [0,1] with a linear IQR extension beyond the training range, CAPPED at +/-oor_cap
    (see OOR_CAP); missing -> 0.5 (median rank). Binary: value as-is; missing -> 0.
    Returns an UNWEIGHTED coordinate matrix; call weight_coords() to apply SHAP weights.
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if binary_mask[j] or fitted[j] is None:
            out[:, j] = np.where(np.isfinite(col), col, 0.0)
            continue
        f = fitted[j]
        c = np.interp(col, f['xs'], f['ps'])  # clips out-of-range to [0,1]
        below = col < f['tmin']
        above = col > f['tmax']
        # linear in IQR units, then saturate so a heavy tail cannot dominate the distance
        ext_lo = np.clip((col - f['tmin']) / f['iqr'], -oor_cap, 0.0)     # in [-cap, 0]
        ext_hi = np.clip(1.0 + (col - f['tmax']) / f['iqr'], 1.0, 1.0 + oor_cap)
        c = np.where(below, ext_lo, c)
        c = np.where(above, ext_hi, c)
        out[:, j] = np.where(np.isfinite(col), c, 0.5)                    # missing -> median rank
    return out


def weight_coords(coords, weights):
    """Apply the mean|SHAP| importance weights so they enter the Euclidean distance."""
    return np.asarray(coords, dtype=np.float64) * weights


# --- z-scale coordinate (literal M&P) -- kept ONLY so the calibration can show that
#     rank-CDF DI predicts OOF degradation better than the raw-z DI (validates the choice).
def fit_zscale(X_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def transform_zscale(arr, mu, sd):
    z = (np.asarray(arr, dtype=np.float64) - mu) / sd
    return np.where(np.isfinite(z), z, 0.0)


def shap_weights(model, X_train_df):
    """mean|SHAP| feature weights (TreeSHAP, tree_path_dependent), normalized to sum 1.

    tree_path_dependent needs no background and handles NaN via the tree default paths --
    the project's canonical importance currency (T25/T41).
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_train_df)
    if isinstance(sv, list):  # some shap versions return a per-class list for binary
        sv = sv[-1]
    mean_abs = np.abs(sv).mean(axis=0)
    if mean_abs.sum() <= 0:
        raise SystemExit("all-zero SHAP importances -- cannot weight the AOA")
    return mean_abs / mean_abs.sum()


# ======================================================================================
# Data loading (shared shape with predict.py so the AOA lands on the same pixels).
# ======================================================================================
def load_model_and_features():
    model_path = MODELS / 'model.json'
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    with open(model_path) as f:
        names = json.load(f)['learner']['feature_names']
    return model, names, model_path


def load_training(names):
    """Training feature matrix in model order + coords + labels. NaN preserved."""
    clean = pd.read_csv(DATA / 'features_clean.csv')
    missing = [c for c in names if c not in clean.columns]
    if missing:
        raise SystemExit(f"features_clean.csv is missing model columns: {missing[:5]} ...")
    X_df = clean[names]
    X = X_df.to_numpy(dtype=np.float64)
    lat = clean['Latitude'].to_numpy()
    lon = clean['Longitude'].to_numpy()
    y = clean['Class'].to_numpy()  # 0 = Abrupt (majority), 1 = Non-abrupt (minority)
    return X_df, X, lat, lon, y


def cv_folds(lat, lon):
    blocks = assign_blocks(lat, lon, method=BLOCK_METHOD, cell_km=OPERATIVE_CELL_KM)
    return list(buffered_block_folds(lat, lon, blocks, n_splits=N_SPLITS,
                                     buffer_km=BUFFER_KM, seed=CV_SEED))


def cv_training_di(zw_train, lat, lon, dbar):
    """CV training DI: each point's DI using only OTHER-fold points as neighbours."""
    folds = cv_folds(lat, lon)
    n = len(zw_train)
    di = np.full(n, np.nan)
    for train_idx, test_idx in folds:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        d = nearest_distance(zw_train[test_idx], zw_train[train_idx])
        di[test_idx] = d / dbar
    return di[np.isfinite(di)]


def boxplot_fence(di):
    q25, q75 = np.percentile(di, [25, 75])
    return q75 + 1.5 * (q75 - q25), q25, q75


# ======================================================================================
# Main pipeline
# ======================================================================================
def main():
    print("=" * 80)
    print("AREA OF APPLICABILITY (AOA) -- reliability layer [T21/G18]")
    print("=" * 80)

    model, names, model_path = load_model_and_features()
    n_features = len(names)
    print(f"\nModel: {model_path}  ({n_features} features)")

    X_df, X_train, lat, lon, _y = load_training(names)
    n_train = len(X_train)
    print(f"Training points: {n_train:,}")

    binary_mask = detect_binary(X_df)
    print(f"Binary one-hot features (passed through 0/1): {int(binary_mask.sum())}")

    print("\nComputing mean|SHAP| feature weights (TreeSHAP on training set)...")
    weights = shap_weights(model, X_df)
    top = np.argsort(weights)[::-1][:8]
    print("  Top weighted features:")
    for i in top:
        print(f"    {weights[i]:.4f}  {names[i]}")

    # --- rank-CDF coordinate: fit on training, weight by SHAP ---------------------------
    print("\nFitting rank->training-CDF coordinate (continuous features)...")
    fitted = fit_rank_cdf(X_train, binary_mask)
    zw_train = weight_coords(transform_rank_cdf(X_train, fitted, binary_mask), weights)

    print("Computing dbar (mean pairwise training distance)...")
    dbar = mean_pairwise_distance(zw_train)
    print(f"  dbar = {dbar:.6f}")

    # --- CV training DI + box-plot fence (fallback / comparison) ------------------------
    print("\nDeriving CV training-DI distribution "
          f"({BLOCK_METHOD} {OPERATIVE_CELL_KM} km, {N_SPLITS} folds, buffer {BUFFER_KM} km)...")
    di_train = cv_training_di(zw_train, lat, lon, dbar)
    fence, q25, q75 = boxplot_fence(di_train)
    print(f"  train DI: median={np.median(di_train):.4f}  Q75={q75:.4f}  IQR={q75 - q25:.4f}")
    print(f"  box-plot fence (Q75 + 1.5*IQR) = {fence:.4f}")

    # --- threshold: calibration-anchored if available, else the fence -------------------
    threshold, thr_rule, thr_source, thr_extra = resolve_threshold(fence)
    print(f"\nAOA threshold = {threshold:.4f}  [{thr_source}]")
    print(f"  rule: {thr_rule}")

    # --- grid ---------------------------------------------------------------------------
    print("\nLoading prediction datacube...")
    ds = xr.open_dataset(DATA / 'prediction_data.nc')
    feature_stack = ds['feature_stack'].values  # (y, x, feature)
    dataset_feature_names = ds['feature'].values.tolist()
    y_size, x_size, _ = feature_stack.shape
    n_pixels = y_size * x_size
    if dataset_feature_names != names:
        idx = [dataset_feature_names.index(n) for n in names]
        feature_stack = feature_stack[:, :, idx]
        print("  reordered datacube features to model order")
    default_value = ds.attrs.get('default_value', -9999)
    grid = feature_stack.reshape(n_pixels, n_features).astype(np.float64)
    grid = np.where(grid == default_value, np.nan, grid)

    # Obu permafrost-domain mask + >=1-feature guard -- IDENTICAL to predict.py [T20].
    if 'longitude' not in ds.coords or 'latitude' not in ds.coords:
        raise SystemExit("prediction_data.nc lacks longitude/latitude coords (rebuild datacube).")
    lon2d = ds['longitude'].values
    lat2d = ds['latitude'].values
    perprob = local_rasters.sample_points(
        local_rasters.OBU_TIF, lon2d.ravel(), lat2d.ravel()).reshape(n_pixels)
    in_domain = perprob > 0
    has_evidence = (~np.isnan(grid)).sum(axis=1) >= 1
    valid = in_domain & has_evidence
    print(f"  valid (in-domain AND >=1 feature) pixels: {int(valid.sum()):,} of {n_pixels:,}")

    valid_idx = np.flatnonzero(valid)
    if SMOKE:
        rng = np.random.default_rng(0)
        valid_idx = np.sort(rng.choice(valid_idx, size=min(50000, len(valid_idx)), replace=False))
        print(f"  [SMOKE] scoring {len(valid_idx):,} sampled valid pixels")

    # --- grid DI + AOA flag + nearest-neighbour index (for drivers) ---------------------
    print("\nComputing grid dissimilarity index (nearest training distance / dbar)...")
    zw_grid = weight_coords(transform_rank_cdf(grid[valid_idx], fitted, binary_mask), weights)
    d_grid, nn_idx = nearest_distance(zw_grid, zw_train, return_index=True)
    di_grid = d_grid / dbar
    inside = di_grid <= threshold

    n_scored = len(valid_idx)
    n_inside = int(inside.sum())
    print(f"  grid DI: median={np.nanmedian(di_grid):.4f}  max={np.nanmax(di_grid):.4f}")
    print(f"  INSIDE AOA (reliable):     {n_inside:,} ({n_inside / n_scored * 100:.1f}%)")
    print(f"  OUTSIDE AOA (extrapolate): {n_scored - n_inside:,} "
          f"({(n_scored - n_inside) / n_scored * 100:.1f}%)")

    # --- drivers of extrapolation: per-feature share of the OUTSIDE-cell distance -------
    drivers = extrapolation_drivers(zw_grid, zw_train, nn_idx, inside, names)
    print("\nDrivers of extrapolation (share of squared distance over OUTSIDE-AOA cells):")
    print(f"  max single-feature share = {drivers['max_share'] * 100:.1f}%  "
          f"({drivers['max_feature']})")
    for name, share in drivers['top']:
        print(f"    {share * 100:5.1f}%  {name}")

    # --- save reliability raster --------------------------------------------------------
    DI = np.full(n_pixels, np.nan)
    AOA = np.full(n_pixels, np.nan)
    DI[valid_idx] = di_grid
    AOA[valid_idx] = inside.astype(float)
    DI2d = DI.reshape(y_size, x_size)
    AOA2d = AOA.reshape(y_size, x_size)

    attrs = {
        'description': 'Area-of-Applicability reliability layer [T21/G18], Meyer & Pebesma 2021',
        'method': ('Importance-weighted dissimilarity index over a rank->training-CDF '
                   'coordinate (continuous features mapped to empirical training-CDF rank '
                   'in [0,1] with linear IQR extension beyond range; binaries 0/1), '
                   'weighted by mean|SHAP| (sum=1); DI = nearest-train-distance / dbar.'),
        'metric': 'rank_cdf',
        'why_not_literal_MP': ('Raw standardized-z distance is dominated by a few heavy-'
                               'tailed features (one feature up to 93% of the distance); '
                               'rank-CDF is bounded so no feature dominates (max share '
                               f'{drivers["max_share"] * 100:.1f}%). Departs from literal '
                               'M&P; justified by the DI-vs-CV-performance calibration.'),
        'weight_source': 'mean|SHAP| from operative all-data model.json (normalized to sum 1)',
        'nan_handling': ('continuous missing -> rank 0.5 (median); binary missing -> 0 '
                         '(absent category); off-domain -> NaN'),
        'dbar_mean_pairwise_train_distance': float(dbar),
        'dbar_note': 'dbar cancels out of the in/out flag; it only sets the DI absolute scale.',
        'aoa_threshold': float(threshold),
        'threshold_source': thr_source,
        'threshold_rule': thr_rule,
        'boxplot_fence_rank_cdf': float(fence),
        'cv_protocol': f'{BLOCK_METHOD} {OPERATIVE_CELL_KM}km, {N_SPLITS} folds, '
                       f'buffer {BUFFER_KM}km, seed {CV_SEED}',
        'inside_aoa_meaning': '1 = inside AOA (reliable); 0 = outside (extrapolating); NaN = off-domain',
        'extrapolation_drivers': '; '.join(f'{n} {s * 100:.1f}%' for n, s in drivers['top']),
        'calibration_caveat': ('The DI-vs-performance curve is measured only over the DI '
                               'range the biased training sample itself spans; the region '
                               'far beyond the sample is uncertain by definition -- the AOA '
                               'flags it, the calibration cannot score it.'),
        'note': 'Separate reliability layer -- NOT folded into susceptibility.nc (T20 design).',
    }
    attrs.update(thr_extra)
    out_ds = xr.Dataset(
        {'DI': (['y', 'x'], DI2d), 'inside_aoa': (['y', 'x'], AOA2d)},
        coords={'x': ds.coords['x'], 'y': ds.coords['y'],
                'longitude': ds.coords['longitude'], 'latitude': ds.coords['latitude']},
        attrs=attrs,
    )
    aoa_path = DATA / 'aoa.nc'
    out_ds.to_netcdf(aoa_path)
    print(f"\nSaved reliability raster: {aoa_path}")

    # --- maps ---------------------------------------------------------------------------
    aoa_map, di_map = save_maps(AOA2d, DI2d, lon2d, lat2d, threshold)

    print("\n" + "=" * 80)
    print("AOA COMPLETE")
    print("=" * 80)
    for p in (aoa_path, aoa_map, di_map):
        print(f"  - {p}")


def resolve_threshold(fence):
    """Pick the AOA threshold: the calibration-anchored value if present (metric must
    match rank_cdf), else the box-plot fence fallback. Returns
    (threshold, rule_str, source_str, extra_attrs_dict)."""
    if THRESHOLD_JSON.exists():
        cal = json.loads(THRESHOLD_JSON.read_text())
        if cal.get('metric') == 'rank_cdf' and np.isfinite(cal.get('threshold', np.nan)):
            extra = {
                'threshold_prevalence_floor': float(cal.get('prevalence_floor', np.nan)),
                'threshold_calibration_note': cal.get('note', ''),
                'threshold_boxplot_fence_from_calibration': float(cal.get('boxplot_fence', np.nan)),
            }
            return (float(cal['threshold']), cal.get('rule', 'calibration'),
                    'calibration (aoa_threshold.json)', extra)
        print(f"  WARNING: {THRESHOLD_JSON.name} present but metric != rank_cdf; "
              "using box-plot fence.")
    else:
        print(f"  NOTE: {THRESHOLD_JSON.name} not found -- run diagnostics/aoa_calibration.py "
              "to anchor the threshold to CV performance. Falling back to the box-plot fence.")
    return (float(fence), 'Q75 + 1.5*IQR of CV training DI (same-fold neighbours excluded)',
            'box-plot fence fallback', {})


def extrapolation_drivers(zw_grid, zw_train, nn_idx, inside, names, chunk=200000):
    """Per-feature share of the squared distance to the nearest training point, summed
    over the OUTSIDE-AOA grid cells. Interpretable "what makes these cells novel?" -- and
    the max share is the check that rank-CDF is genuinely multivariate (no feature dominates).
    """
    out = ~inside
    if out.sum() == 0:
        return {'top': [], 'max_share': 0.0, 'max_feature': 'none', 'feat_share': None}
    g = zw_grid[out]
    ref = zw_train[nn_idx[out]]
    feat_total = np.zeros(zw_grid.shape[1], dtype=np.float64)
    for s in range(0, len(g), chunk):
        diff = g[s:s + chunk] - ref[s:s + chunk]
        feat_total += (diff * diff).sum(axis=0)
    total = feat_total.sum()
    share = feat_total / total if total > 0 else feat_total
    order = np.argsort(share)[::-1]
    return {
        'top': [(names[i], float(share[i])) for i in order[:8]],
        'max_share': float(share[order[0]]),
        'max_feature': names[order[0]],
        'feat_share': share,
    }


def save_maps(AOA2d, DI2d, lon2d, lat2d, threshold):
    from matplotlib.colors import ListedColormap
    _ok = (np.isfinite(lon2d) & (np.abs(lon2d) <= 180)
           & np.isfinite(lat2d) & (np.abs(lat2d) <= 90))
    extent = [float(lon2d[_ok].min()), float(lon2d[_ok].max()),
              float(lat2d[_ok].min()), float(lat2d[_ok].max())]
    OUTPUT.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(np.flipud(AOA2d), extent=extent, aspect='auto', origin='lower',
                   interpolation='nearest', cmap=ListedColormap(['#c0392b', '#2e8b57']),
                   vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Outside AOA\n(extrapolating)', 'Inside AOA\n(reliable)'])
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('Area of Applicability — where the statewide map is a reliable interpolation\n'
                 f'(rank-CDF DI, Meyer & Pebesma 2021; threshold = {threshold:.2f})',
                 fontsize=13, fontweight='bold')
    aoa_map = OUTPUT / 'aoa_map.png'
    plt.savefig(aoa_map, dpi=600, bbox_inches='tight')
    print(f"Saved AOA map: {aoa_map}")

    fig2, ax2 = plt.subplots(figsize=(14, 10))
    im2 = ax2.imshow(np.flipud(DI2d), extent=extent, aspect='auto', origin='lower',
                     interpolation='nearest', cmap='magma_r',
                     vmax=float(np.nanpercentile(DI2d, 99)))
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Dissimilarity index (0 = at a training point)', rotation=270, labelpad=20)
    ax2.contour(np.flipud((DI2d > threshold).astype(float)), levels=[0.5],
                extent=extent, colors='cyan', linewidths=0.4)
    ax2.set_xlabel('Longitude (°E)', fontsize=12)
    ax2.set_ylabel('Latitude (°N)', fontsize=12)
    ax2.set_title('AOA dissimilarity index (rank-CDF; cyan = AOA boundary)',
                  fontsize=13, fontweight='bold')
    di_map = OUTPUT / 'aoa_di_map.png'
    plt.savefig(di_map, dpi=600, bbox_inches='tight')
    print(f"Saved DI map: {di_map}")
    plt.close('all')
    return aoa_map, di_map


if __name__ == '__main__':
    main()
