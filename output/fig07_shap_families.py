"""Figure 7 — SHAP indicator-family importance, colored by thematic domain.

A single magnitude panel: the 22 emergent feature *families* (feature-space
Spearman clusters, NOT SHAP-space — the anti-circularity point), importance-
sorted, each bar as tall as its family's mean |Σ member SHAP| (margin) and
filled with its thematic-domain color. Two annotations per bar: the family's
share of summed family importance, and its member count n (so redundancy is
visible — a 5% family built from 7 correlated members is a different claim than
a 3% family that is one column).

The honest result the figure communicates is a MULTI-FACTOR signal. The 22 rows
must not be eye-summed into a "climate ≈ N%" claim: families remain mutually
correlated (|rho| up to 0.55), so a cross-family sum is illegitimate. The
lower-right inset gives the ONLY legitimate cross-family subtotal — the eight
thematic-domain aggregates, computed one level up from the same per-point SHAP —
and shows Relief and Temperature co-leading at ~25% each, neither dominating.

Direction is deliberately NOT shown here (the old panel (b) violins are dropped):
the sample is ~94% Abrupt, so nearly every family leans warm relative to that
abrupt-heavy average — prevalence, not a per-indicator claim. The prior-free
reading is the log-evidence index and the Fig 8 dependence shapes (where SHAP
crosses zero). Land Cover — the one genuinely bidirectional family — gets its own
per-class figure.

Domain palette and the feature/family -> domain maps live in shap_domains.py, a
single source of truth Fig 9 (the dominance map) imports too, so the two figures
cannot drift. The dendrogram documenting family construction is in the Supplement
(output/shap_family_dendrogram.png).

Data:
  output/shap_grouped_matrix.npz   — per-point grouped-SHAP matrix (family bars),
                                     columns importance-sorted; from models/shap_groups.py.
  output/shap_mechanism_cache.npz  — per-feature OOF SHAP (inset domain aggregates);
                                     same OOF SHAP, one level finer, so the two are consistent.

Writes output/07_shap_families.{pdf,png}. Pure plotting; regenerate the caches
(multi-minute OOF fold-refit TreeSHAP) if the model or feature set changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import figstyle          # noqa: E402
import shap_domains as sd  # noqa: E402

GROUPED = _HERE / "shap_grouped_matrix.npz"
FAMILIES_JSON = _HERE / "shap_families.json"
MECH = _HERE / "shap_mechanism_cache.npz"

INSET_TITLE = "Feature-level thematic aggregate"

# Family bar labels are the emergent-cluster labels from the SHAP cache; a few
# are relabeled here for the reader (presentation only — the grouping in
# models/shap_groups.py and the cache keys are untouched). Anything not listed
# renders with its cache label.
DISPLAY_LABEL = {
    "Annual / dry-season temperature": "Annual Mean Temperature",
    "Isothermality / precip seasonality": "Seasonality",
    "Summer warmth": "Summer temperature",
    "Soil organic / fertility": "Soil organic content",
    "Eastness": "Aspect (eastness)",
    "Northness": "Aspect (northness)",
}


def load_families():
    """(labels, importance, n_members) with columns importance-sorted descending.

    Member counts are read from shap_families.json (keyed by family label) rather
    than hardcoded, so a regrouping in models/shap_groups.py flows through without
    editing this script.
    """
    if not GROUPED.exists():
        raise FileNotFoundError(
            f"{GROUPED} not found — run `poetry run python models/shap_groups.py` first.")
    d = np.load(GROUPED, allow_pickle=True)
    labels, importance = list(d["labels"]), d["importance"].astype(float)
    n_by_label = {f["label"]: f["n_members"]
                  for f in json.loads(FAMILIES_JSON.read_text())["families"]}
    missing = [l for l in labels if l not in n_by_label]
    if missing:
        raise KeyError(f"no n_members in {FAMILIES_JSON.name} for families: {missing}")
    n_members = [n_by_label[l] for l in labels]
    return labels, importance, n_members


def load_domain_shares():
    """Domain aggregates computed live from per-feature SHAP (share-sorted desc)."""
    if not MECH.exists():
        raise FileNotFoundError(f"{MECH} not found — needed for the domain-aggregate inset.")
    c = np.load(MECH, allow_pickle=True)
    return sd.domain_aggregates(c["values"], list(c["feature_names"]))


def family_panel(ax, labels, importance, n_members):
    """Horizontal domain-colored family bars, importance-sorted (top = strongest)."""
    n = len(labels)
    ys = np.arange(n)[::-1]
    colors = [sd.DOMAIN_COLORS[sd.FAMILY_DOMAIN[l]] for l in labels]
    frac = importance / importance.sum() * 100.0

    ax.barh(ys, importance, height=0.74, color=colors, zorder=2)
    for y, imp, f, k in zip(ys, importance, frac, n_members):
        pct = f"{f:.0f}%" if f >= 0.5 else "<1%"
        ax.annotate(f"{pct}   n={k}",
                    xy=(imp, y), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6.4, color=figstyle.MUTED)

    ax.set_yticks(ys)
    ax.set_yticklabels([DISPLAY_LABEL.get(l, l) for l in labels], fontsize=7.3)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, importance.max() * 1.24)
    ax.set_xlabel("Mean absolute contribution to log-evidence", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)


def aggregate_inset(ax, shares):
    """Lower-right inset: 8 domain-aggregate bars, descending, in domain colors.

    Doubles as the domain color key for the family bars (each bar labeled in its
    own color), so no separate legend is needed.
    """
    doms = list(shares)                     # already share-sorted descending
    vals = np.array([shares[d]["share"] * 100 for d in doms])
    ys = np.arange(len(doms))[::-1]
    colors = [sd.DOMAIN_COLORS[d] for d in doms]

    ax.barh(ys, vals, height=0.72, color=colors, zorder=2)
    # share % just past each bar tip — always on white, so readable over any fill
    for y, v in zip(ys, vals):
        ax.annotate(f"{v:.1f}%", xy=(v, y), xytext=(3, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=6.2, color=figstyle.INK, zorder=4)
    ax.set_xlim(0, vals.max() * 1.16)
    ax.set_ylim(-0.7, len(doms) - 0.3)
    ax.set_xticks([])
    # domain names in a left gutter (the color key), never over a dark bar
    ax.set_yticks(ys)
    ax.set_yticklabels(doms, fontsize=6.2, color=figstyle.INK)
    ax.tick_params(axis="y", length=0)
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(INSET_TITLE, fontsize=7.3, color=figstyle.INK, loc="left", pad=4,
                 fontweight="bold")


def main():
    figstyle.use()
    labels, importance, n_members = load_families()
    shares = load_domain_shares()

    fig = figstyle.figure("onehalf", height=7.7, subplots=False)
    ax = fig.add_axes([0.30, 0.065, 0.685, 0.915])
    family_panel(ax, labels, importance, n_members)

    # inset in the lower-right whitespace left by the short tail bars; shifted far
    # enough right that its longest gutter label clears the family-annotation column
    ax_in = ax.inset_axes([0.50, 0.045, 0.48, 0.35])
    aggregate_inset(ax_in, shares)

    figstyle.save(fig, "07_shap_families")
    plt.close(fig)

    print(f"wrote output/07_shap_families.{{pdf,png}}  ({len(labels)} families)")
    print("\nfamily order (importance desc):")
    for l, i, k in zip(labels, importance, n_members):
        print(f"  {i:7.4f}  n={k:2d}  {sd.FAMILY_DOMAIN[l]:26s}  {l}")
    print("\ndomain aggregates (inset):")
    for d, rec in shares.items():
        print(f"  {rec['share']*100:5.1f}%  {d}")
    print("\npalette:")
    for d, c in sd.DOMAIN_COLORS.items():
        print(f"  {c}  {d}")


if __name__ == "__main__":
    main()
