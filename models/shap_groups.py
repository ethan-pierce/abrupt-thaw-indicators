"""Grouped (emergent-family) SHAP for the operative thaw-mode model (TASKS T41).

Purpose (a): a de-cluttered, geoscientist-legible indicator-FAMILY importance ranking, so
SHAP credit is not split across the ~70 partly-redundant feature columns. Families emerge
data-drivenly from feature-space redundancy, then their SHAP is recombined by additivity.
Family *interpretation* is post-hoc (SCOPE Headline C); this is NOT the mechanism-vs-lake-
proxy analysis (deferred separately).

Design (grill 2026-07-15; see memory t41-grouped-shap-design):
- Grouping basis: FEATURE-space (Spearman on feature values), not SHAP-space -- a tree
  scatters credit erratically across near-duplicate columns, so SHAP-space can fail to
  group them; feature-space groups by shared information regardless of how the model split
  the credit (25/44 continuous cols have a |Spearman|>0.8 partner here).
- Distance: 1 - |Spearman| (absolute) -- anti-correlated columns are still redundant
  (fire pair rho=-1.00; thermal continentality spans +-0.94); signed distance would
  fragment them.
- Linkage: complete -- a cut at distance t means every within-family pair has |rho| >= 1-t;
  also the linkage that best resists |rho|-induced chaining.
- Cut: the natural GAP in the merge-height sequence (auto-detected, emergent -- not a round
  number), where tightly-redundant families stop merging and only weak relations remain.
- Categoricals: collapse one-hots to their SOURCE (Land Cover, Vegetation Mode) by summing
  member SHAP; a lone binary (Yedoma) stays standalone. A one-hot family IS one variable,
  so this is definitional redundancy -- the same "recombine split credit" logic.
- Grouped contribution per point = SUM of signed member SHAP (exact additivity). Grouped
  global importance = mean over points of |sum|. Abrupt-oriented (positive => toward Abrupt),
  inherited from pooled_oof_shap.

Reuses the canonical OOF-SHAP machinery from shap_values.py (per-fold refit + held-out
TreeSHAP), so grouping is the only thing added. Run after the operative model / feature set
is final (post-T23 lock). SHAP_GROUPS_SMOKE=1 subsamples for a fast correctness check.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block on plt.show()
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT
# Reuse the canonical inputs + OOF-SHAP machinery (no re-implementation, exact parity).
from shap_values import (load_inputs, load_cv_config, load_selected_hparams,
                         pooled_oof_shap, SMOKE_HPARAMS)

# Fast smoke config for correctness checks (SHAP_GROUPS_SMOKE=1); no effect on real runs.
SMOKE = bool(os.environ.get('SHAP_GROUPS_SMOKE'))
SMOKE_N = 1500
SMOKE_SPLITS = 3

# Gap-cut search band, in distance = 1 - |Spearman| (i.e. |rho| in [0.40, 0.85]). The
# largest gap between consecutive merge heights INSIDE this band is the emergent cut; the
# band guards against trivial gaps near the root (all-merged) or in the noise floor.
GAP_BAND = (0.15, 0.60)

# One-hot families collapsed to their categorical source (definitional redundancy). Any
# remaining lone binary column (e.g. Yedoma) becomes its own standalone family.
CATEGORICAL_PREFIXES = ('Land Cover', 'Vegetation Mode')

# Manuscript family names for the emergent MULTI-member continuous families, keyed by the
# exact member set (order-independent). Settled in the grill design (memory
# t41-grouped-shap-design) and confirmed by the two curation calls (2026-07-16): the
# alpine-relief four stay fused (all pairwise |rho| 0.69-0.81, no 2+2 seam) and Trend in SWE
# stays in thermal continentality (|rho| 0.66-0.76 to the thermal block vs -0.29 to Trend in
# temperature). A cluster whose membership isn't a key here keeps the auto-tag + a warning,
# so a future re-cluster surfaces instead of silently mislabelling.
MANUSCRIPT_LABELS = {
    frozenset({'Elevation', 'Slope', 'Height Above Nearest Drainage',
               'Mean Annual SWE'}): 'Alpine relief',
    frozenset({'Annual Mean Temperature',
               'Mean Temperature of Driest Quarter'}): 'Annual / dry-season temperature',
    frozenset({'Mean Diurnal Range', 'Temperature Seasonality',
               'Min Temperature of Coldest Month', 'Temperature Annual Range',
               'Mean Temperature of Coldest Quarter', 'Trend in SWE'}): 'Thermal continentality',
    frozenset({'Isothermality',
               'Precipitation Seasonality'}): 'Isothermality / precip seasonality',
    frozenset({'Annual Precipitation', 'Precipitation of Wettest Month',
               'Precipitation of Driest Month', 'Precipitation of Wettest Quarter',
               'Precipitation of Driest Quarter', 'Precipitation of Warmest Quarter',
               'Precipitation of Coldest Quarter'}): 'Precipitation amount',
    frozenset({'Max Temperature of Warmest Month', 'Mean Temperature of Wettest Quarter',
               'Mean Temperature of Warmest Quarter'}): 'Summer warmth',
    frozenset({'Sand (0-30 cm)', 'Sand (30-200 cm)'}): 'Sand fraction',
    frozenset({'Soil Organic Carbon (0-30 cm)', 'Soil Organic Carbon (30-200 cm)',
               'Nitrogen (0-30 cm)', 'Nitrogen (30-200 cm)'}): 'Soil organic / fertility',
    frozenset({'Clay (0-30 cm)', 'Clay (30-200 cm)'}): 'Clay fraction',
    frozenset({'Bulk Density (0-30 cm)', 'Bulk Density (30-200 cm)'}): 'Bulk density',
    frozenset({'Time Since Last Fire', 'Burn Count'}): 'Fire history',
}


# --------------------------------------------------------------------------
# emergent families (feature-space clustering + categorical collapse)
# --------------------------------------------------------------------------
def split_columns(X):
    """Partition columns into continuous vs one-hot (values subset of {0,1})."""
    onehot = [c for c in X.columns if set(pd.unique(X[c].dropna())) <= {0, 1}]
    cont = [c for c in X.columns if c not in onehot]
    return cont, onehot


def continuous_linkage(X, cont):
    """Complete-linkage tree over continuous columns, distance = 1 - |Spearman|.

    A constant/degenerate column (Spearman NaN, e.g. a rare feature in a smoke subsample) is
    treated as unrelated (|rho| -> 0, distance -> 1) so the linkage never sees a NaN.
    """
    S = np.nan_to_num(X[cont].corr(method='spearman').abs().values, nan=0.0)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0  # enforce exact symmetry for squareform
    return linkage(squareform(D, checks=False), method='complete')


def choose_gap_threshold(Z, band=GAP_BAND):
    """Cut at the midpoint of the largest gap between consecutive merge heights in `band`.

    Emergent, data-driven: the merge sequence has a natural discontinuity where tightly-
    redundant families stop forming and only weakly-related columns remain. Returns
    (threshold, gap_size, lo_height, hi_height).
    """
    h = np.sort(Z[:, 2])
    lo, hi = band
    hb = h[(h >= lo) & (h <= hi)]
    if len(hb) < 2:
        return (lo + hi) / 2.0, 0.0, lo, hi  # degenerate: fall back to band midpoint
    gaps = np.diff(hb)
    k = int(np.argmax(gaps))
    lo_h, hi_h = float(hb[k]), float(hb[k + 1])
    return (lo_h + hi_h) / 2.0, float(gaps[k]), lo_h, hi_h


def build_families(X, threshold=None):
    """Emergent continuous families (complete-linkage gap cut) + collapsed categoricals.

    Returns (families, meta): `families` maps a provisional key -> [member columns]
    (continuous keys are `cont_<id>`, renamed to legible labels after importance is known);
    `meta` carries the linkage, chosen threshold/gap, and the signed correlation for
    within-family diagnostics.
    """
    cont, onehot = split_columns(X)
    Z = continuous_linkage(X, cont)
    if threshold is None:
        threshold, gap, lo_h, hi_h = choose_gap_threshold(Z)
    else:
        gap = lo_h = hi_h = None
    labels = fcluster(Z, t=threshold, criterion='distance')

    families = {}
    for cl in sorted(set(labels)):
        families[f"cont_{cl}"] = [cont[i] for i in range(len(cont)) if labels[i] == cl]

    # Categoricals collapsed to source; any leftover lone binary stands alone.
    claimed = set()
    for prefix in CATEGORICAL_PREFIXES:
        members = [c for c in onehot if c.startswith(prefix)]
        if members:
            families[prefix] = members
            claimed.update(members)
    for c in onehot:
        if c not in claimed:
            families[c] = [c]

    meta = {'threshold': threshold, 'gap': gap, 'gap_lo': lo_h, 'gap_hi': hi_h,
            'linkage': Z, 'continuous': cont, 'onehot': onehot,
            'signed_corr': X[cont].corr(method='spearman')}
    return families, meta


# --------------------------------------------------------------------------
# grouped SHAP (exact additivity)
# --------------------------------------------------------------------------
def grouped_shap_matrix(expl, families):
    """Sum signed member SHAP per family -> (names, (n_points, n_families)).

    Additivity of SHAP makes a family's per-point contribution to the margin EXACTLY the sum
    of its members' SHAP. Values are already Abrupt-oriented (positive => toward Abrupt).
    """
    cols = list(expl.feature_names)
    idx = {c: i for i, c in enumerate(cols)}
    names = list(families.keys())
    G = np.zeros((expl.values.shape[0], len(names)))
    for j, name in enumerate(names):
        member_idx = [idx[c] for c in families[name] if c in idx]
        G[:, j] = expl.values[:, member_idx].sum(axis=1)
    return names, G


def display_label(key, members, expl):
    """Legible label for a family.

    Multi-member continuous families use the settled manuscript name (MANUSCRIPT_LABELS,
    keyed by member set); an unmapped multi-member cluster falls back to its top-importance
    member + "(+k)" and warns, so a re-cluster surfaces rather than mislabelling silently.
    Singletons keep their own column name; categoricals are named by source + class count.
    """
    if key in ('Land Cover', 'Vegetation Mode'):
        return f"{key} ({len(members)} classes)"
    if len(members) == 1:
        return members[0]
    manuscript = MANUSCRIPT_LABELS.get(frozenset(members))
    if manuscript is not None:
        return manuscript
    cols = list(expl.feature_names)
    idx = {c: i for i, c in enumerate(cols)}
    imp = {m: np.mean(np.abs(expl.values[:, idx[m]])) for m in members if m in idx}
    top = max(imp, key=imp.get)
    print(f"[warn] no manuscript label for family {sorted(members)}; using auto-tag "
          f"'{top} (+{len(members) - 1})'. Update MANUSCRIPT_LABELS if the cut changed.")
    return f"{top} (+{len(members) - 1})"


def within_family_min_abs_rho(members, signed_corr):
    """Smallest |Spearman| among continuous family members (None if not applicable)."""
    present = [m for m in members if m in signed_corr.columns]
    if len(present) < 2:
        return None
    sub = signed_corr.loc[present, present].values
    off = sub[np.triu_indices(len(present), 1)]
    return float(np.abs(off).min())


# --------------------------------------------------------------------------
# figures + report
# --------------------------------------------------------------------------
def plot_dendrogram(meta, families, labels_by_key, out_dir):
    """Emergent continuous-family dendrogram: gap-cut line plus a named band per family.

    Every continuous family (singletons included) gets an alternating-shade horizontal band
    spanning the full width, so its name reads straight across from its member leaves. The
    band, not a distant bracket, carries the leaf -> family tie. Categorical one-hot families
    (Land Cover, Vegetation Mode) collapse by source and are not leaves; the caption names
    them. Metadata (linkage, cut value, within-family rho floor) lives in the caption too.
    """
    Z, cont, t = meta['linkage'], meta['continuous'], meta['threshold']
    fig, ax = plt.subplots(figsize=(15, max(6, 0.30 * len(cont))))
    dn = dendrogram(Z, labels=cont, orientation='right', color_threshold=t,
                    leaf_font_size=8, ax=ax)

    # scipy lays leaves out at data-y = 5, 15, 25, ... in `ivl` (bottom-to-top) order.
    ypos = {lab: 5 + 10 * i for i, lab in enumerate(dn['ivl'])}
    bands = []
    for key, members in families.items():
        if not key.startswith('cont_'):
            continue  # categoricals collapse by source -> not leaves in this tree
        ys = [ypos[m] for m in members if m in ypos]
        if ys:
            bands.append((min(ys), max(ys), labels_by_key[key]))
    bands.sort()
    for i, (lo, hi, name) in enumerate(bands):
        if i % 2 == 0:  # every other family shaded, so adjacent bands stay distinct
            ax.axhspan(lo - 5, hi + 5, facecolor='0.93', edgecolor='none', zorder=0)
        ax.text(1.02, (lo + hi) / 2, name, va='center', ha='left',
                fontsize=9, fontweight='bold', color='0.15')

    ax.axvline(t, color='k', ls='--', lw=1, zorder=1)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("distance = 1 − |Spearman|")
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_bounds(0.0, 1.0)  # axis line stops at the last tick, not the labels
    fig.subplots_adjust(left=0.21, right=0.99, top=0.99, bottom=0.07)
    plt.savefig(out_dir / 'shap_family_dendrogram.pdf')  # vector, canonical for the manuscript
    plt.savefig(out_dir / 'shap_family_dendrogram.png', dpi=300)
    plt.close()


def plot_grouped_importance(order, labels, importance, out_dir):
    """Horizontal bar of grouped mean|sum SHAP|, families ranked (top at the top)."""
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(order))))
    y = np.arange(len(order))[::-1]
    ax.barh(y, importance[order], color='#4477aa')
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax.set_xlabel("Grouped importance:  mean over points of | sum of member SHAP |  (margin)")
    ax.set_title("Indicator-family SHAP importance (Abrupt-oriented)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_grouped_importance.png', dpi=300)
    plt.close()


def plot_grouped_contribution_box(order, labels, G, out_dir):
    """Per-family box of signed grouped SHAP (direction + spread); 0 = neutral."""
    top = order[:min(20, len(order))]
    data = [G[:, i] for i in top][::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(top))))
    ax.boxplot(data, vert=False, showfliers=False, whis=(5, 95))
    ax.set_yticklabels([labels[i] for i in top][::-1], fontsize=8)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel("Grouped SHAP (margin): >0 favours Abrupt, <0 favours Non-abrupt")
    ax.set_title("Per-family contribution distribution (top 20 by importance)")
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_grouped_contribution_box.png', dpi=300)
    plt.close()


def write_families_json(order, keys, labels, families, importance, meta, n_points, out_dir):
    """Machine-readable record: memberships, importances, within-family min|rho|, cut."""
    total = float(importance.sum())
    rec = {
        'method': {
            'grouping_basis': 'feature-space Spearman (values)',
            'distance': '1 - |Spearman|',
            'linkage': 'complete',
            'cut_threshold': meta['threshold'],
            'cut_rho_floor': 1 - meta['threshold'],
            'gap_size': meta['gap'],
            'gap_between_heights': [meta['gap_lo'], meta['gap_hi']],
            'categorical_collapse': list(CATEGORICAL_PREFIXES) + ['(lone binaries standalone)'],
            'importance_metric': 'mean over points of |sum of signed member SHAP| (margin, Abrupt-oriented)',
            'n_points_scored': int(n_points),
            'smoke': SMOKE,
        },
        'families': [
            {
                'rank': r + 1,
                'label': labels[i],
                'members': families[keys[i]],
                'n_members': len(families[keys[i]]),
                'importance': float(importance[i]),
                'importance_frac': float(importance[i] / total) if total else None,
                'within_family_min_abs_spearman': within_family_min_abs_rho(
                    families[keys[i]], meta['signed_corr']),
            }
            for r, i in enumerate(order)
        ],
    }
    (out_dir / 'shap_families.json').write_text(json.dumps(rec, indent=2))


def write_grouped_matrix(order, labels, G, importance, out_dir):
    """Persist the per-point grouped-SHAP matrix so figures can plot distributions.

    The JSON keeps only summary importances; the per-point signed contributions
    (needed for the Fig 6 violins) are otherwise discarded when this script ends.
    Store them column-reordered by descending importance so downstream plotting is
    a pure load (mirrors the diagnostics -> cached-artifact -> figure pattern used
    by Fig 5). Labels are saved in the SAME (importance-sorted) column order as G.
    """
    order = np.asarray(order, dtype=int)
    np.savez(
        out_dir / 'shap_grouped_matrix.npz',
        G=G[:, order].astype(np.float32),                 # (n_points, n_families), importance-sorted
        labels=np.array([labels[i] for i in order], dtype=object),
        importance=importance[order].astype(np.float64),
    )


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

    print(f"Grouped OOF SHAP: {len(y)} points | {X.shape[1]} features | "
          f"operative cell {cfg['operative_cell_km']} km | buffer {cfg['buffer_km']} km | "
          f"{n_splits} folds | hyperparameters {hparams}")

    expl, scored = pooled_oof_shap(
        X, y, lat, lon,
        cell_km=cfg['operative_cell_km'], buffer_km=cfg['buffer_km'],
        n_splits=n_splits, seed=cfg['seeds']['CV_SEED'], hparams=hparams,
    )

    # Families are defined on the SAME scored feature matrix the SHAP was computed on.
    X_scored = X[scored].reset_index(drop=True)
    families, meta = build_families(X_scored)
    print(f"Cut at dist {meta['threshold']:.3f} (|rho| >= {1 - meta['threshold']:.2f}); "
          f"largest gap {meta['gap']:.3f} between heights "
          f"[{meta['gap_lo']:.3f}, {meta['gap_hi']:.3f}] -> {len(families)} families")

    keys, G = grouped_shap_matrix(expl, families)
    labels = [display_label(k, families[k], expl) for k in keys]
    importance = np.mean(np.abs(G), axis=0)
    order = list(np.argsort(importance)[::-1])

    # Smoke results are non-authoritative (subsampled): keep them out of the real output/
    # so they can never be mistaken for the deliverable.
    out_dir = OUTPUT / '_smoke' if SMOKE else OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dendrogram(meta, families, dict(zip(keys, labels)), out_dir)
    plot_grouped_importance(order, labels, importance, out_dir)
    plot_grouped_contribution_box(order, labels, G, out_dir)
    write_families_json(order, keys, labels, families, importance, meta,
                        expl.values.shape[0], out_dir)
    write_grouped_matrix(order, labels, G, importance, out_dir)

    print("\nTop indicator families by grouped SHAP importance:")
    total = importance.sum()
    for r, i in enumerate(order[:15]):
        frac = importance[i] / total * 100 if total else 0.0
        rho = within_family_min_abs_rho(families[keys[i]], meta['signed_corr'])
        rho_s = f", min|rho|={rho:.2f}" if rho is not None else ""
        print(f"  {r + 1:>2}. {labels[i]:<42} {importance[i]:.4f} ({frac:4.1f}%) "
              f"[{len(families[keys[i]])} feat{rho_s}]")
    print(f"\nWrote family figures + shap_families.json to {out_dir} "
          f"({int(scored.sum())} points explained out-of-fold)")
    print("NOTE: multi-member families carry their settled manuscript labels "
          "(MANUSCRIPT_LABELS); an unmapped cluster would fall back to an auto-tag and warn. "
          "Both curation calls resolved (2026-07-16): alpine-relief four stay fused, Trend in "
          "SWE stays in thermal continentality (see memory t41-grouped-shap-design).")


if __name__ == '__main__':
    main()
