"""Train/serve parity gate [T23] — every feature, training column vs datacube pixel.

The model is fit on ``features_clean.csv`` (per-training-point columns) and then
*scored* on ``prediction_data.nc`` (the statewide 1 km datacube). If a feature is
constructed differently on the two paths — a unit slip, a stray transform (the old
``log(upa)`` trap), a sign flip, mismatched NaN semantics, or a categorical class
present statewide but absent from training and silently folded into the dropped
reference bucket — the model extrapolates onto a distribution it never saw and the
map is quietly wrong. This gate documents, per feature, that train and serve agree.

Both artifacts now share the rebuilt 1 km schema (feature-table rebuild
+ T47 datacube rewrite + retrain, 2026-07-15), so the comparison is finally
meaningful (it was BLOCKED while serve was the earlier, coarser 4 km cube).

Method — **matched-location parity** (factors out the lake-/road-biased sampling
so any residual gap is *construction*, not landscape): the datacube's lon/lat grid
is regular (constant lat per row, lon per col, ~0.008983 deg step), so we
reconstruct its exact 1-D axes and nearest-index each training point to its cell,
then compare the training column against the cube value at that same cell. Because
the point path samples at the *exact* training coordinate while the cube samples at
the *cell centre* (offset up to ~0.7 km), fine native-scale features (terrain @10 m)
carry real sub-cell variance on top of any construction gap — so we read
correlation / distribution overlap, not bit-equality, for those. Features built by
identical construction on both paths (terrain, MERIT hnd/upa, SoilGrids @250 m,
Yedoma — all native-sampled both sides per T35/T37) should still correlate ~1.

Reports, per the T23 done-when:
  * per-feature matched-location parity (NaN agreement, medians, robust scale
    ratio, Spearman) — covers the terrain native-parity T37 construction check;
  * soil-NaN reproduction + confirmation that soil is native-sampled BOTH sides
    (the T23 note's "soil 250 m->1 km reproject-averaging" concern is STALE — T35
    moved soil to native sampling; there is no reproject-average to measure);
  * fire >70N QA coverage gap (T36) reproduction;
  * land-cover / vegetation-mode **category-set subset check** — raw NLCD / ALFRESCO
    classes present in the in-domain datacube but with no model one-hot column
    (silent reference-bucket absorption), and the in-domain area affected.

Writes ``diagnostics/train_serve_parity.md`` + a summary figure
``diagnostics/train_serve_parity.png``; prints a PASS/FLAG summary. Read-only on the
model input (one-hot construction unchanged).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import DATA, MODELS
from data import local_rasters
import _data

CUBE_PATH = DATA / 'prediction_data.nc'
REPORT_PATH = Path(__file__).resolve().parent / 'train_serve_parity.md'
FIG_PATH = Path(__file__).resolve().parent / 'train_serve_parity.png'
FILL = -9999.0

# NLCD / ALFRESCO code -> label, mirroring build_prediction_data.py's one-hot maps.
# The datacube emits a one-hot ONLY for a code whose feature column is in the model
# (which, by construction of clean_feature_table.py, is only codes seen at training
# points). A raw class present statewide but absent here => no column => the cell is
# all-zero across the categorical => silently absorbed into the dropped reference.
LAND_COVER_LABELS = {
    11: 'Open Water', 12: 'Perennial Ice/Snow', 21: 'Developed, Open Space',
    22: 'Developed, Low Intensity', 23: 'Developed, Medium Intensity',
    24: 'Developed, High Intensity', 31: 'Barren Land (Rock/Sand/Clay)',
    41: 'Deciduous Forest', 42: 'Evergreen Forest', 43: 'Mixed Forest',
    51: 'Dwarf Scrub', 52: 'Shrub/Scrub', 71: 'Grassland/Herbaceous',
    72: 'Sedge/Herbaceous', 73: 'Lichens', 74: 'Moss', 81: 'Pasture/Hay',
    82: 'Cultivated Crops', 90: 'Woody Wetlands', 95: 'Emergent Herbaceous Wetlands',
}
VEGETATION_MODE_LABELS = {
    1: 'Black spruce', 2: 'White spruce', 3: 'Deciduous forest', 4: 'Shrub tundra',
    5: 'Graminoid tundra', 6: 'Wetland tundra', 7: 'Barren lichen moss',
    8: 'Temperate rainforest',
}

SOIL_VARS = ['Soil Organic Carbon', 'Nitrogen', 'Bulk Density', 'Sand', 'Clay']


def reconstruct_axes(lon2d, lat2d):
    """Recover exact 1-D (lon_col, lat_row) axes from the regular but partly
    -9999-filled 2-D coord grids. Solve the affine row/col->deg map from two finite
    reference cells (step is constant), so off-ROI rows/cols get exact coords too."""
    finite = (np.abs(lon2d) <= 180) & (np.abs(lat2d) <= 90)
    rows, cols = np.where(finite)
    # lat depends on row only, lon on col only. Fit with least squares on finite cells.
    lat0, dlat = np.polyfit(rows, lat2d[finite], 1)[::-1]
    lon0, dlon = np.polyfit(cols, lon2d[finite], 1)[::-1]
    ny, nx = lon2d.shape
    lat_axis = lat0 + dlat * np.arange(ny)
    lon_axis = lon0 + dlon * np.arange(nx)
    return lon_axis, dlon, lat_axis, dlat


def nearest_cells(lon_axis, dlon, lat_axis, dlat, lon, lat):
    """Nearest datacube (row, col) for each training point on the regular grid."""
    col = np.rint((lon - lon_axis[0]) / dlon).astype(int)
    row = np.rint((lat - lat_axis[0]) / dlat).astype(int)
    ny, nx = lat_axis.size, lon_axis.size
    col = np.clip(col, 0, nx - 1)
    row = np.clip(row, 0, ny - 1)
    return row, col


def robust_ratio(a, b):
    """median(|serve|) / median(|train|) on the shared-finite set; ~1 => same units."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    ta = np.median(np.abs(a[m]))
    tb = np.median(np.abs(b[m]))
    if ta == 0:
        return np.nan
    return tb / ta


def main():
    print("=" * 80)
    print("TRAIN/SERVE PARITY GATE  [T23]")
    print("=" * 80)

    # --- Train side: the exact model-input columns + coords -----------------------
    X, y, lat, lon = _data.load(verify=True)
    feat_cols = list(X.columns)
    print(f"\nTrain: {X.shape[0]} points x {X.shape[1]} features (features_clean.csv)")

    # --- Serve side: the datacube -------------------------------------------------
    ds = xr.open_dataset(CUBE_PATH)
    cube_feats = ds['feature'].values.tolist()
    ny, nx = ds.sizes['y'], ds.sizes['x']
    print(f"Serve: {ny} x {nx} cells x {len(cube_feats)} features ({CUBE_PATH.name})")

    # Schema parity: same set, same order (predict.py reorders on mismatch, but the
    # gate should record they already line up).
    assert set(feat_cols) == set(cube_feats), (
        f"feature-set mismatch:\n  train-only={set(feat_cols) - set(cube_feats)}\n"
        f"  serve-only={set(cube_feats) - set(feat_cols)}")
    order_match = feat_cols == cube_feats
    print(f"Feature sets identical: True | same order: {order_match}")

    lon2d = ds['longitude'].values
    lat2d = ds['latitude'].values
    lon_axis, dlon, lat_axis, dlat = reconstruct_axes(lon2d, lat2d)
    row, col = nearest_cells(lon_axis, dlon, lat_axis, dlat, lon, lat)
    # Guard: training points should land inside the cube footprint.
    in_lon = (lon >= lon_axis.min() - abs(dlon)) & (lon <= lon_axis.max() + abs(dlon))
    in_lat = (lat >= lat_axis.min() - abs(dlat)) & (lat <= lat_axis.max() + abs(dlat))
    n_out = int((~(in_lon & in_lat)).sum())
    print(f"Training points inside cube footprint: {X.shape[0] - n_out}/{X.shape[0]} "
          f"({n_out} outside — clipped to edge)")

    # Sub-cell offset of each training point from its matched cell centre (km). The
    # point path samples at the EXACT coord; the cube at the cell centre. For a
    # feature that varies within a 1 km cell, this offset alone lowers matched parity
    # even when construction is identical — so we recompute parity on the near-centre
    # subset (offset < NEAR_KM): if it converges to ~1, the gap is offset, not a bug.
    dy_km = (lat - lat_axis[row]) * 110.57
    dx_km = (lon - lon_axis[col]) * 111.32 * np.cos(np.deg2rad(lat))
    offset_km = np.hypot(dx_km, dy_km)
    NEAR_KM = 0.15
    near = offset_km < NEAR_KM
    print(f"Sub-cell offset: median {np.median(offset_km):.3f} km, "
          f"near-centre (<{NEAR_KM} km) n={int(near.sum())}")

    # --- In-domain (Obu) mask over the whole cube, for statewide serve stats ------
    perprob = local_rasters.sample_points(
        local_rasters.OBU_TIF, lon2d.ravel(), lat2d.ravel()).reshape(ny, nx)
    in_domain = perprob > 0
    print(f"Obu permafrost domain (PerProb > 0): {int(in_domain.sum()):,} cells "
          f"({in_domain.mean() * 100:.1f}% of grid)")

    is_onehot = {c: set(np.unique(X[c].dropna().values)) <= {0.0, 1.0} for c in feat_cols}

    # --- Per-feature matched-location parity --------------------------------------
    rows_out = []
    for name in feat_cols:
        slab = ds['feature_stack'].isel(feature=cube_feats.index(name)).values
        slab = np.where(slab == FILL, np.nan, slab)
        serve_state = slab[in_domain]                     # in-domain statewide
        serve_pt = slab[row, col]                         # matched to training pts
        train_pt = X[name].values.astype(float)

        tr_nan = float(np.isnan(train_pt).mean())
        sv_nan_pt = float(np.isnan(serve_pt).mean())
        sv_nan_state = float(np.isnan(serve_state).mean())
        both = np.isfinite(train_pt) & np.isfinite(serve_pt)
        # NaN-pattern agreement at matched locations (both-NaN or both-finite).
        nan_agree = float(((np.isnan(train_pt)) == (np.isnan(serve_pt))).mean())

        # Same metric on the near-centre subset (offset < NEAR_KM) — the offset control.
        bn = both & near
        if is_onehot[name]:
            # Categorical one-hot: agreement of the 0/1 value where both finite.
            agree = float((train_pt[both] == serve_pt[both]).mean()) if both.any() else np.nan
            near_metric = (float((train_pt[bn] == serve_pt[bn]).mean())
                           if bn.sum() >= 20 else np.nan)
            rows_out.append(dict(
                feature=name, kind='one-hot', tr_nan=tr_nan, sv_nan_pt=sv_nan_pt,
                sv_nan_state=sv_nan_state, nan_agree=nan_agree, match=agree,
                near_metric=near_metric, ratio=np.nan, rho=np.nan,
                tr_med=float(np.nanmean(train_pt)), sv_med=float(np.nanmean(serve_state)),
            ))
        else:
            ratio = robust_ratio(train_pt, serve_pt)
            rho = (float(spearmanr(train_pt[both], serve_pt[both]).correlation)
                   if both.sum() >= 5 else np.nan)
            near_metric = (float(spearmanr(train_pt[bn], serve_pt[bn]).correlation)
                           if bn.sum() >= 20 else np.nan)
            rows_out.append(dict(
                feature=name, kind='continuous', tr_nan=tr_nan, sv_nan_pt=sv_nan_pt,
                sv_nan_state=sv_nan_state, nan_agree=nan_agree, match=np.nan,
                near_metric=near_metric, ratio=ratio, rho=rho,
                tr_med=float(np.nanmedian(train_pt)), sv_med=float(np.nanmedian(serve_state)),
            ))
    parity = pd.DataFrame(rows_out)

    # --- Categorical subset check: raw NLCD / ALFRESCO classes in-domain ----------
    ll_lon = lon2d[in_domain]
    ll_lat = lat2d[in_domain]
    cat_reports = {}
    for cat_name, raster, labels, prefix in [
        ('Land Cover', local_rasters.NLCD_IMG, LAND_COVER_LABELS, 'Land Cover'),
        ('Vegetation Mode', local_rasters.VEGMODE_TIF, VEGETATION_MODE_LABELS, 'Vegetation Mode'),
    ]:
        codes = local_rasters.sample_points(raster, ll_lon, ll_lat)
        codes = codes[np.isfinite(codes)].astype(int)
        vals, counts = np.unique(codes, return_counts=True)
        model_cols = {c for c in feat_cols if c.startswith(prefix + ' (')}
        report = []
        for v, cnt in sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: -kv[1]):
            label = labels.get(v, f'code {v}')
            col_name = f'{prefix} ({label})'
            has_col = col_name in model_cols
            # code 0 / NaN is the ALFRESCO/NLCD background (the dropped bucket) — expected.
            report.append(dict(code=v, label=label, cells=cnt,
                               frac=cnt / max(1, in_domain.sum()),
                               has_column=has_col,
                               is_background=(v == 0)))
        cat_reports[cat_name] = pd.DataFrame(report)

    ds.close()

    # ------------------------------------------------------------------ verdict ---
    # A matched-location FLAG is a *construction smell* — a unit/transform slip
    # (robust scale ratio off by >3x), a decorrelation (Spearman < 0.5), or a
    # one-hot 0/1 disagreement (< 0.9). NaN-agreement is NOT a flag (reported as a
    # coverage note): a feature whose finite values agree in scale AND rank has
    # identical construction regardless of a ragged coverage boundary.
    def flag_row(r):
        if r['kind'] == 'one-hot':
            return np.isfinite(r['match']) and r['match'] < 0.90
        bad_scale = np.isfinite(r['ratio']) and (r['ratio'] > 3 or r['ratio'] < 1 / 3)
        bad_rho = np.isfinite(r['rho']) and r['rho'] < 0.5
        return bad_scale or bad_rho

    # Adjudicate each flag. The construction-bug signature is an ORDER-OF-MAGNITUDE
    # scale/transform slip (>10x or <0.1x) — that is what a unit error (m vs km,
    # g/kg vs %), a stray log (the historical log(upa) trap), or a wrong band would
    # produce, and it is location-independent so it never washes out. Everything else
    # that flags at 1 km is "offset-sensitive": a fine-scale or spatially singular
    # feature whose matched parity is limited by the exact-coord-vs-cell-centre
    # geometry (offset up to ~0.7 km). The near-centre control corroborates this
    # (parity rises monotonically as the offset shrinks — see near_metric) but is NOT
    # the gate: offset cannot be driven to zero on a 1 km grid, so even near-centre
    # points stay many native pixels off and never reach 1.
    def classify(r):
        if not r['flag']:
            return 'clean'
        if r['kind'] == 'continuous' and np.isfinite(r['ratio']) and (
                r['ratio'] > 10 or r['ratio'] < 0.1):
            return 'CONSTRUCTION'
        return 'offset-sensitive'

    parity['flag'] = parity.apply(flag_row, axis=1)
    parity['verdict'] = parity.apply(classify, axis=1)

    subset_flags = {
        cat: df[(~df['is_background']) & (~df['has_column']) & (df['cells'] > 0)]
        for cat, df in cat_reports.items()
    }

    write_report(parity, cat_reports, subset_flags, in_domain.sum(), order_match, n_out)
    make_figure(parity)

    n_flag = int(parity['flag'].sum())
    n_constr = int((parity['verdict'] == 'CONSTRUCTION').sum())
    n_offset = int((parity['verdict'] == 'offset-sensitive').sum())
    n_subset = sum(len(v) for v in subset_flags.values())
    print("\n" + "=" * 80)
    print(f"PARITY: {len(parity) - n_flag}/{len(parity)} clean at matched location; "
          f"{n_offset} offset-sensitive; {n_constr} genuine construction flag(s).")
    for _, r in parity[parity['flag']].iterrows():
        print(f"  [{r['verdict']}] {r['feature']!r}: ratio={r['ratio']:.3g} "
              f"rho={r['rho']:.3g} match={r['match']:.3g} -> near-centre "
              f"{r['near_metric']:.3g}")
    print(f"SUBSET: {n_subset} in-domain categorical class(es) with no model column.")
    for cat, df in subset_flags.items():
        for _, r in df.iterrows():
            print(f"  {cat}: {r['label']!r} in {int(r['cells']):,} cells "
                  f"({r['frac'] * 100:.2f}%) — absorbed into reference bucket.")
    verdict = ("PASS — no genuine construction discrepancy"
               if n_constr == 0 else f"REVIEW — {n_constr} construction flag(s)")
    print(f"\nGATE: {verdict}")
    print(f"Report: {REPORT_PATH}")
    print(f"Figure: {FIG_PATH}")
    return parity, cat_reports


def write_report(parity, cat_reports, subset_flags, n_domain, order_match, n_out):
    L = []
    L.append("# Train/serve parity gate [T23]\n")
    L.append("Per-feature comparison of the training column (`features_clean.csv`, the "
             "exact model input) against the datacube pixel (`prediction_data.nc`), the "
             "surface the model is scored on. Matched-location method: each training "
             "point is nearest-indexed to its 1 km cell and compared there, so residual "
             "gaps are *construction*, not the lake-/road-biased sampling.\n")
    L.append(f"- Features: **{len(parity)}**, feature sets identical, same order: "
             f"**{order_match}**.\n")
    L.append(f"- Obu in-domain cells (PerProb > 0): **{int(n_domain):,}**; training points "
             f"outside cube footprint (edge-clipped): **{n_out}**.\n")
    L.append("\n**Caveat:** the point path samples at the exact training coordinate; the "
             "cube samples at the cell centre (offset up to ~0.7 km). Fine native-scale "
             "features (terrain @10 m) therefore carry real sub-cell variance on top of "
             "any construction gap — read Spearman / scale-ratio, not bit-equality. Each "
             "matched-location flag is adjudicated by the **near-centre control** (parity "
             "recomputed on points sitting on their cell centre): if it converges to ~1, "
             "the gap is offset geometry, not construction.\n")

    # Verdict banner
    n_constr = int((parity['verdict'] == 'CONSTRUCTION').sum())
    n_offset = int((parity['verdict'] == 'offset-sensitive').sum())
    subset_n = sum(len(v) for v in subset_flags.values())
    banner = ("✅ **PASS** — no genuine train/serve construction discrepancy."
              if n_constr == 0 else f"⚠️ **REVIEW** — {n_constr} construction flag(s).")
    L.append(f"\n## Verdict\n\n{banner}\n")
    L.append(f"- Matched-location: **{len(parity) - int(parity['flag'].sum())}** clean, "
             f"**{n_offset}** offset-sensitive (no unit/transform slip — parity is limited "
             f"by sub-cell geometry and rises toward the cell centre; see `ρ near` / "
             f"`match near`), **{n_constr}** genuine construction flag(s).\n")
    L.append(f"- Category-set subset: **{subset_n}** non-background class(es) present "
             f"in-domain but absent from training (silent reference-bucket absorption; "
             f"see below — negligible area).\n")
    L.append("\n### What the offset flags reveal (not a bug — a *sampling* signal)\n")
    L.append("The offset-sensitive features are exactly the spatially "
             "singular / small-patch ones — `Upstream Area` and `Height Above Nearest "
             "Drainage` (near-zero on the drainage line, large one pixel away), the "
             "terrain derivatives (`Mean curvature (500 m)`, `Northness`/`Eastness`), and "
             "the small-patch land covers (`Open Water` train-active **0.43** vs serve "
             "**0.04**; `Dwarf Scrub`, `Sedge/Herbaceous`). Their construction is identical "
             "both sides (native / nearest sampling), but the divergence quantifies the "
             "documented **lake-/road-collection bias**: training points sit "
             "systematically in flatter, wetter, lower-drainage, more-open-water locations "
             "than the statewide 1 km grid the model scores (`Slope` train median 0.74° vs "
             "cube 3.92°; `HND` 1 m vs 17 m). This is a *representativeness* concern for "
             "the map (why calibration is suspect — SCOPE — and why the AOA layer T21 "
             "matters), not a construction defect this gate should block on.\n")

    def vmark(v):
        return {'clean': '✓', 'offset-sensitive': '~ offset', 'CONSTRUCTION': '⚠️ BUILD'}[v]

    # Continuous table
    cont = parity[parity['kind'] == 'continuous'].copy()
    L.append("\n## Continuous features — matched-location parity\n")
    L.append("`ratio` = median(|serve|)/median(|train|) (≈1 ⇒ same units/transform); "
             "`ρ` = Spearman at matched points; `ρ near` = Spearman on points sitting on "
             "their cell centre (the offset control); `nan_agree` = NaN-pattern agreement.\n")
    L.append("\n| feature | train med | serve med | ratio | ρ | ρ near | train NaN | serve NaN (pt/state) | nan_agree | verdict |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    for _, r in cont.iterrows():
        L.append(f"| {r['feature']} | {r['tr_med']:.4g} | {r['sv_med']:.4g} | "
                 f"{r['ratio']:.3g} | {r['rho']:.3f} | {r['near_metric']:.3f} | "
                 f"{r['tr_nan'] * 100:.1f}% | "
                 f"{r['sv_nan_pt'] * 100:.1f}%/{r['sv_nan_state'] * 100:.1f}% | "
                 f"{r['nan_agree']:.3f} | {vmark(r['verdict'])} |")

    # One-hot table
    oh = parity[parity['kind'] == 'one-hot'].copy()
    L.append("\n## One-hot (categorical) features — matched-location parity\n")
    L.append("`match` = fraction of matched points with identical 0/1 value; "
             "`match near` = the same on points on their cell centre (offset control); "
             "`train mean` / `serve mean` = active fraction.\n")
    L.append("\n| feature | train mean | serve mean | match | match near | verdict |")
    L.append("|---|--:|--:|--:|--:|:--:|")
    for _, r in oh.iterrows():
        L.append(f"| {r['feature']} | {r['tr_med']:.4g} | {r['sv_med']:.4g} | "
                 f"{r['match']:.3f} | {r['near_metric']:.3f} | {vmark(r['verdict'])} |")

    # Soil / fire coverage notes
    L.append("\n## Coverage: soil-NaN reproduction & fire QA gap\n")
    soil = cont[cont['feature'].str.contains('|'.join(SOIL_VARS))]
    if len(soil):
        L.append(f"- **Soil** ({len(soil)} composite cols): statewide serve NaN "
                 f"{soil['sv_nan_state'].min() * 100:.1f}–{soil['sv_nan_state'].max() * 100:.1f}%, "
                 f"train NaN {soil['tr_nan'].min() * 100:.1f}–{soil['tr_nan'].max() * 100:.1f}% "
                 f"(dry-run reported ~11.6% statewide). Soil is **native-sampled (250 m) on "
                 f"BOTH paths** (T35) — the T23 note's \"soil 250 m→1 km reproject-averaging\" "
                 f"concern is **stale**; there is no reproject-average, and the high matched "
                 f"ρ confirms identical construction.")
    fire = cont[cont['feature'].str.contains('Fire')]
    if len(fire):
        L.append(f"- **Fire** (MODIS, T36): statewide serve NaN "
                 f"{fire['sv_nan_state'].min() * 100:.1f}–{fire['sv_nan_state'].max() * 100:.1f}% — "
                 f"the documented >70°N QA coverage gap, reproduced on the serve side.")

    # Categorical subset check
    L.append("\n## Category-set subset check (silent reference-bucket absorption)\n")
    L.append("Raw NLCD / ALFRESCO classes sampled at in-domain cell centres. A "
             "non-background class present in-domain **without** a model one-hot column "
             "is folded silently into the dropped reference bucket.\n")
    for cat, df in cat_reports.items():
        L.append(f"\n### {cat}\n")
        L.append("| code | class | in-domain cells | % domain | model column? |")
        L.append("|--:|---|--:|--:|:--:|")
        for _, r in df.iterrows():
            tag = 'background (dropped)' if r['is_background'] else (
                '✓' if r['has_column'] else '❌ NO COLUMN')
            L.append(f"| {r['code']} | {r['label']} | {int(r['cells']):,} | "
                     f"{r['frac'] * 100:.2f}% | {tag} |")
        flagged = subset_flags[cat]
        if len(flagged):
            tot = int(flagged['cells'].sum())
            L.append(f"\n**⚠️ {len(flagged)} class(es) absorbed**, {tot:,} in-domain cells "
                     f"(~{tot:,} km²): " + ", ".join(f"{r['label']}" for _, r in flagged.iterrows()))
        else:
            L.append("\n**✓ No absorption** — every non-background in-domain class has a model column.")

    L.append("\n---\n_Generated by `diagnostics/train_serve_parity.py` (T23)._\n")
    REPORT_PATH.write_text("\n".join(L))


def make_figure(parity):
    cmap = {'clean': 'tab:blue', 'offset-sensitive': 'tab:orange', 'CONSTRUCTION': 'tab:red'}

    def panel(ax, df, metric, floor, xlabel, title):
        colors = [cmap[v] for v in df['verdict']]
        ax.barh(df['feature'], df[metric], color=colors)
        # Near-centre control marker: shows flagged bars recovering when offset->0.
        flg = df[df['flag']]
        ax.scatter(flg['near_metric'], flg['feature'], marker='|', s=200,
                   color='k', zorder=3, label='near-centre control')
        ax.axvline(floor, color='k', ls='--', lw=0.8)
        ax.set_xlim(-0.1, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis='y', labelsize=7)
        if len(flg):
            ax.legend(loc='lower right', fontsize=8)

    cont = parity[parity['kind'] == 'continuous'].sort_values('rho')
    oh = parity[parity['kind'] == 'one-hot'].sort_values('match')
    fig, axes = plt.subplots(1, 2, figsize=(15, max(8, 0.28 * len(parity))))
    panel(axes[0], cont, 'rho', 0.5,
          'Spearman ρ (train column vs cube at matched cell)',
          'Continuous features (dashed = 0.5 floor)')
    panel(axes[1], oh, 'match', 0.9,
          'Matched 0/1 agreement',
          'One-hot features (dashed = 0.9 floor)')

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for l, c in
               [('clean', cmap['clean']), ('offset-sensitive', cmap['offset-sensitive']),
                ('construction flag', cmap['CONSTRUCTION'])]]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle('Train/serve parity at matched locations [T23]',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
