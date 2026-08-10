"""Single source of truth for SHAP *thematic domain* colors and membership.

Both Figure 6 (family-importance bars + domain-aggregate inset) and Figure 9
(the domain-dominance map) import this module, so the two figures cannot drift
in either the palette or the feature-to-domain assignment.

Two levels of grouping live here:

  * **Domain** — the coarse thematic bucket (Topography, Baseline temperature,
    Snow, …). Nine of them. Colors are the categorical Crameri ``batlowS``
    swatches (CVD-safe; same scientific-colour-map family as ``batlow``/``vik``
    used elsewhere), plus one off-map warm gold for the 9th domain (Summer
    temperature) — batlowS runs out of ΔE≥15-separable swatches at eight. The
    palette is validated (see ``validate()``).

The thermal features split into two map domains: **Baseline temperature** (the
cold-season / annual thermal state — annual mean, dry- and cold-quarter means,
winter minimum, plus the minor seasonal-range and warming-trend passengers) and
**Summer temperature** (the warm-season peak — warmest month/quarter, wettest
quarter). The split is spatially real (verified over the grid): the cold/annual
baseline dominates the maritime-influenced west and south, the summer peak
dominates the continental interior. It follows the emergent-family correlation
structure (the *Summer warmth* family vs. the *Annual-dry-season* + *Thermal
continentality* + *Trend* families), so it is not hand-drawn. The seasonal-range
and trend features never dominate any cell (they push in mixed signs and cancel),
so they ride inside Baseline rather than forming their own inert domain.
  * **Feature -> domain** — every one of the 70 model features maps to exactly one
    domain. Categorical one-hots (``Land Cover (*)``, ``Vegetation Mode (*)``) are
    assigned FIRST, before any name matching, so e.g. "Land Cover (Barren Land
    (Rock/Sand/Clay))" cannot false-match the Soil bucket.

Snow and Seasonality are feature-level domains only: their SWE / isothermality
features sit *inside* the relief and thermal emergent families, so they carry no
family bar but do get a domain aggregate (this is intentional — Fig 6 footnotes
it). The family->domain map (for coloring the 22 family bars) is separate and
covers the 7 domains that surface at family level.

Domain aggregate metric (``domain_aggregates``): mean over points of
|sum over the domain's features of signed SHAP| — the same |Σ member SHAP|
construction used for families, one level up. Shares are each aggregate over the
sum of the eight. Because families remain mutually correlated, these domain
totals are the ONLY legitimate cross-family subtotal; per-family numbers must not
be summed by hand (that is the whole reason the inset exists).

Consumers that store domain codes positionally (e.g. Fig 9's dominance cache,
which argmaxes over ``DOMAIN_ORDER``) MUST regenerate after any change here — the
split reorders ``DOMAIN_ORDER`` and renumbers the codes.
"""

from __future__ import annotations

import numpy as np
from cmcrameri import cm as _cmc
from matplotlib.colors import to_hex

# --------------------------------------------------------------------------- #
# Domains (canonical names — used as dict keys, display labels, and the join key
# Fig 9 matches on; do not rename without updating Fig 9).
# --------------------------------------------------------------------------- #
RELIEF = "Topography"
BASELINE = "Baseline temperature"
SUMMER = "Summer temperature"
SNOW = "Snow (SWE)"
LANDCOVER = "Land Cover and Fire History"
PRECIPITATION = "Precipitation"
SEASONALITY = "Seasonality"
SOIL = "Soil"
YEDOMA = "Yedoma"

# Canonical display order (used only when a stable non-sorted order is wanted;
# the figures sort by share/importance). Summer temperature sits beside Baseline
# temperature as its thermal sibling.
DOMAIN_ORDER = [RELIEF, BASELINE, SUMMER, SNOW, LANDCOVER,
                PRECIPITATION, SEASONALITY, SOIL, YEDOMA]

# --------------------------------------------------------------------------- #
# Palette — Crameri batlowS (categorical). Colors are pulled live from the map
# so they are provably the scientific-colour-map swatches, not hand-typed hex.
# Index -> swatch (batlowS first 8): 0 navy 1 pale-pink 2 olive 3 teal
#   4 peach 5 green 6 blue 7 pink. Assignment is chosen so the two narrative
# co-leaders (Topography, Baseline temperature) are maximally distinct and every
# adjacent inset pair clears a wide CVD margin; see fig07 handoff spec.
# --------------------------------------------------------------------------- #
_S = [to_hex(_cmc.batlowS(i)) for i in range(8)]
# Summer temperature is the 9th domain; batlowS has no further swatch that clears
# the ΔE≥15 CVD floor against the first eight, so it takes an off-map warm gold —
# a thermal sibling of Baseline's peach (both warm, so the eye reads "thermal";
# gold also reads as summer heat), and the only tone found that keeps the whole
# 9-color palette CVD-safe (min ΔE 18.6 across normal + deuteranopia +
# protanopia; see validate()).
_SUMMER_HEX = "#E1A100"
DOMAIN_COLORS = {
    RELIEF: _S[3],              # teal
    BASELINE: _S[4],            # peach (warm — cold-season / annual baseline)
    SUMMER: _SUMMER_HEX,        # gold  (warm — warm-season peak)
    SNOW: _S[6],                # blue  (ice / water)
    LANDCOVER: _S[5],           # green (vegetation)
    PRECIPITATION: _S[0],       # navy  (water)
    SEASONALITY: _S[7],         # pink
    SOIL: _S[2],                # olive (earth)
    YEDOMA: _S[1],              # pale pink (palest -> smallest domain)
}

# --------------------------------------------------------------------------- #
# Feature -> domain. Explicit membership lists (70 features total). Categorical
# one-hots are handled by prefix in feature_domain() BEFORE these are consulted.
# --------------------------------------------------------------------------- #
# Baseline temperature = cold-season / annual thermal state: annual & dry-/cold-
# quarter means and winter minimum (the wins), plus the seasonal-range and trend
# passengers that never dominate a cell but are thermally part of the baseline.
_BASELINE = {
    "Annual Mean Temperature", "Mean Temperature of Driest Quarter",
    "Min Temperature of Coldest Month", "Mean Temperature of Coldest Quarter",
    "Mean Diurnal Range", "Temperature Seasonality", "Temperature Annual Range",
    "Trend in temperature",
}
# Summer temperature = warm-season peak (the *Summer warmth* emergent family).
_SUMMER = {
    "Max Temperature of Warmest Month", "Mean Temperature of Warmest Quarter",
    "Mean Temperature of Wettest Quarter",
}
_PRECIPITATION = {
    "Annual Precipitation", "Precipitation of Wettest Month",
    "Precipitation of Driest Month", "Precipitation of Wettest Quarter",
    "Precipitation of Driest Quarter", "Precipitation of Warmest Quarter",
    "Precipitation of Coldest Quarter", "Trend in precipitation",
}
_SNOW = {"Mean Annual SWE", "Trend in SWE"}
_SEASONALITY = {"Isothermality", "Precipitation Seasonality"}
_RELIEF = {
    "Slope", "Elevation", "Height Above Nearest Drainage", "Mean curvature (500 m)",
    "Mean curvature (2 km)", "Upstream Area", "Eastness", "Northness",
}
_SOIL = {
    "Sand (0-30 cm)", "Sand (30-200 cm)", "Clay (0-30 cm)", "Clay (30-200 cm)",
    "Bulk Density (0-30 cm)", "Bulk Density (30-200 cm)",
    "Soil Organic Carbon (0-30 cm)", "Soil Organic Carbon (30-200 cm)",
    "Nitrogen (0-30 cm)", "Nitrogen (30-200 cm)",
}
_FIRE = {"Flammability Index", "Time Since Last Fire", "Burn Count"}
_YEDOMA = {"Yedoma"}

_EXPLICIT = {}
for _dom, _members in [
    (BASELINE, _BASELINE), (SUMMER, _SUMMER),
    (PRECIPITATION, _PRECIPITATION), (SNOW, _SNOW),
    (SEASONALITY, _SEASONALITY), (RELIEF, _RELIEF), (SOIL, _SOIL),
    (LANDCOVER, _FIRE), (YEDOMA, _YEDOMA),
]:
    for _m in _members:
        _EXPLICIT[_m] = _dom

# --------------------------------------------------------------------------- #
# Family -> domain (for coloring the 22 emergent-family bars). Keys are the
# family labels as written in output/shap_grouped_matrix.npz. Seven domains
# surface at family level; Snow and Seasonality... Seasonality does surface
# (Isothermality / precip seasonality); Snow does not (its two SWE features live
# inside the Relief and Temperature families).
# --------------------------------------------------------------------------- #
FAMILY_DOMAIN = {
    "Alpine relief": RELIEF,
    "Mean curvature (500 m)": RELIEF,
    "Mean curvature (2 km)": RELIEF,
    "Upstream Area": RELIEF,
    "Eastness": RELIEF,
    "Northness": RELIEF,
    "Annual / dry-season temperature": BASELINE,
    "Summer warmth": SUMMER,
    "Thermal continentality": BASELINE,
    "Trend in temperature": BASELINE,
    "Precipitation amount": PRECIPITATION,
    "Trend in precipitation": PRECIPITATION,
    "Isothermality / precip seasonality": SEASONALITY,
    "Sand fraction": SOIL,
    "Clay fraction": SOIL,
    "Bulk density": SOIL,
    "Soil organic / fertility": SOIL,
    "Land Cover (18 classes)": LANDCOVER,
    "Vegetation Mode (7 classes)": LANDCOVER,
    "Flammability Index": LANDCOVER,
    "Fire history": LANDCOVER,
    "Yedoma": YEDOMA,
}


def feature_domain(name: str) -> str:
    """Return the thematic domain for a single feature name.

    One-hot categoricals are matched by prefix FIRST, so a land-cover class whose
    name contains a soil-ish word ("Barren Land (Rock/Sand/Clay)") is never
    misfiled. Raises KeyError for an unrecognized feature (fail loud — a silent
    fallthrough would quietly corrupt the domain aggregates).
    """
    if name.startswith("Land Cover (") or name.startswith("Vegetation Mode ("):
        return LANDCOVER
    try:
        return _EXPLICIT[name]
    except KeyError:
        raise KeyError(f"feature {name!r} is not assigned to any domain") from None


def domain_of_features(feature_names) -> np.ndarray:
    """Vector of domain labels aligned to `feature_names`."""
    return np.array([feature_domain(f) for f in feature_names], dtype=object)


def domain_aggregates(values: np.ndarray, feature_names) -> dict:
    """Compute the honest per-domain SHAP aggregate and share.

    values : (n_points, n_features) signed per-feature SHAP (margin units).
    Returns {domain: {"aggregate": float, "share": float, "n_features": int}}
    for every domain present, sorted by aggregate descending.
    """
    values = np.asarray(values, float)
    doms = domain_of_features(feature_names)
    aggregates, counts = {}, {}
    for dom in DOMAIN_ORDER:
        cols = np.where(doms == dom)[0]
        if cols.size == 0:
            continue
        # sum signed SHAP within the domain per point, then mean |.| over points
        aggregates[dom] = float(np.mean(np.abs(values[:, cols].sum(axis=1))))
        counts[dom] = int(cols.size)
    total = sum(aggregates.values())
    ordered = sorted(aggregates, key=aggregates.get, reverse=True)
    return {
        dom: {"aggregate": aggregates[dom],
              "share": aggregates[dom] / total,
              "n_features": counts[dom]}
        for dom in ordered
    }


def validate(feature_names=None) -> None:
    """Self-check: palette is CVD-safe and (if given) every feature is assigned."""
    import figstyle
    de = figstyle.assert_cvd_safe(list(DOMAIN_COLORS.values()), min_de=15,
                                  name="SHAP domain palette (batlowS)")
    print("domain palette CVD ΔE:", {k: round(v, 1) for k, v in de.items()})
    for dom, c in DOMAIN_COLORS.items():
        print(f"  {c}  {dom}")
    if feature_names is not None:
        doms = domain_of_features(feature_names)
        from collections import Counter
        print("feature counts per domain:", dict(Counter(doms)))
        assert len(doms) == len(feature_names)
    print("OK — shap_domains is consistent.")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    c = np.load(Path(__file__).resolve().parent / "shap_mechanism_cache.npz",
                allow_pickle=True)
    fn = list(c["feature_names"])
    validate(fn)
    print("\ndomain aggregates (live from mechanism cache):")
    agg = domain_aggregates(c["values"], fn)
    for dom, d in agg.items():
        print(f"  {d['share']*100:5.1f}%  {dom:26s} agg={d['aggregate']:.4f} "
              f"n={d['n_features']}")
