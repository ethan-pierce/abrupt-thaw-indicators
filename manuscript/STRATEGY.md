# Manuscript strategy — handoff

Decisions from the 2026-07-19/20 grilling session on framing, findings, and evidence
for the draft manuscript. Pipeline/data/modeling are vetted (see `diagnostics/FINDINGS.md`);
goal is a draft on a short timeline, avoiding substantial revision. This doc is the
argument spine + committed decisions; figures fall out of the chain, not vice versa.

## Target & spine

- **Venue: Earth's Future** (AGU, full length — deliberately, so the load-bearing
  caveats have room; a GRL letter would force cutting the very material that armors the map).
- **Spine: map-led.** The statewide **abrupt-thaw-*mode* susceptibility surface** is the
  headline. Novelty hook = predicting thaw **mode** (abrupt vs non-abrupt), which no
  incumbent does (they predict occurrence or stage). Not "continuous vs categorical" —
  that fight is settled.

## Headline result & its framing constraints

- **~a quarter (25.6%) of the in-AOA permafrost domain has geospatial features more
  consistent with abrupt than non-abrupt thaw.** (Computed: 710,882 of 2,773,804 in-AOA
  cells with log-evidence > 0.)
- Anchored **only** to the prior-free **log-evidence = 0** boundary (likelihood ratio = 1).
  Any other threshold is arbitrary and indefensible — do not introduce one.
- **AOA-restricted**: AOA masks only 2.7% of the valid domain (good story — applicable
  nearly everywhere it has data). Report the masked fraction; compute the % inside AOA only.
- **Language discipline**: "features more consistent with abrupt thaw" / feature-consistency —
  **never** "% susceptible", "% will thaw abruptly", or probability. The index is log-evidence,
  not a calibrated probability, not a class.
- **Preempt the "a quarter is high" reaction**: feature-consistency (predisposition) is
  expected to exceed *realized occurrence*; we are not forecasting area that will thaw.
  This is the strongest argument for the log-evidence framing over a probability.
- Median in-AOA log-evidence = −2.50 (typical cell favors non-abrupt) — the coherent
  counterweight; abrupt is the minority mode.
- TODO: confirm datacube cell size to convert 25.6% → absolute km² for the abstract.

## Credibility framing

- **Headline metric: spatial-block-CV AUC-PR ≈ 0.85** (operative 10 km, repeated-CV 0.852 ± 0.011),
  **≈15× the 0.0574 prevalence floor**, reported with across-partition spread (σ ≈ 0.01–0.03 over 20 reshuffles).
- **AUC-ROC (~0.98–0.99): mention once, label minority-insensitive, never headline.**
- **Do not quote the random-split 0.90** — inflated by spatial leakage (findings flag #1/#2).
- **Logistic regression (0.78) is the *floor*, not a rival.** Frame the +0.076 ± 0.019 margin
  as robustness: the signal is strong enough a linear model recovers most of it; XGBoost adds
  a small stable nonlinear margin. **XGBoost is justified by native NaN handling** (SoilGrids
  ~16% NaN; MODIS fire QA gap >70°N) **+ SHAP interpretability**, not by beating logistic.

## Indicators — descriptive, not causal

- Framing verbatim: **"these variables help *distinguish* abrupt from non-abrupt thaw —
  not cause it, not result from it."** Discrimination, not causation.
- Reserve mechanistic language only where physical priors back it (alpine relief →
  ground-ice/drainage; temperature → thermal state); label the rest as associations.
- **No partialling / spatial-CV-survival analysis** (the deferred mechanism-vs-proxy test).
  Rationale: sampling bias is likely inseparable from mechanistic causality, so (b) would
  probably fail to disentangle and risks revision. Lean instead on **leave-region-out
  (AUC-PR 0.54 @ 251 km, ~9× floor)** as the "not just location-proxy" evidence.
- Grouped SHAP families (Abrupt-oriented, from OOF fold-refit SHAP): Alpine relief 23%,
  Annual/dry-season temperature 16%, Land Cover 12%, Thermal continentality 9%, … .

## Argument chain (figures fall out of this)

| Link | Claim | Evidence |
| --- | --- | --- |
| **L1** | Mode matters & is unmapped (abrupt drives outsized C feedback; no incumbent predicts mode) | literature positioning |
| **L2a** | Mode is learnable out-of-sample, above chance | spatial-CV AUC-PR 0.85 ≈ 15× floor |
| **L2b** | It's real, not an artifact | shuffle→chance; dummies at floor; no leakage passthrough; only 4 contradictory-label pairs (noise doesn't bound separation) |
| **L2c** | Not model-specific | logistic floor 0.78; XGBoost earns place via NaN handling + SHAP |
| **L3** | Signal generalizes across space (not memorized neighborhoods) | graceful decay to 0.54 @ 251 km; block-size ladder. *(The link the lake/road bias most threatens.)* |
| **L4a** | Why an index, not a probability | sample ≠ landscape prevalence; prior-free log-evidence |
| **L4b** | AOA bounds where it's trustworthy | AOA mask (2.7% flagged) |
| **L4c** | Representativeness stated honestly | parity gate: training points flatter/wetter/lower-drainage than statewide grid |
| **L5** | Headline number — 25.6% abrupt-favoring; physically coherent (ice-rich lowlands, NW) | the map |
| **L6a** | Ranked family importance | grouped SHAP |
| **L6b** | Direction — which way each family pushes (distinguishes *how*) | SHAP direction/contribution |
| **L6c** | Drivers vary in space (which family dominates where) | → L7 SHAP-dominance map |
| **L7** | Once believed, what the map reveals | ecoregion breakdown; SHAP-dominance map; incumbent contrast |

## L7 downstream additions (committed)

1. **SHAP-dominance map** (highest value): per-cell TreeSHAP over ~2.85M in-AOA cells with
   `models/model.json`, mapped as the dominant grouped family per cell. Descriptive only
   (all-data model, restrict display to AOA). **Doubles as the proxy rebuttal** — if drivers
   vary regionally, the map isn't one smooth spatial trend.
2. **Ecoregion breakdown**: abrupt-favoring fraction by **physiographic ecoregion** (EPA
   Level III Ecoregions of Alaska), **dropping ecoregions below a permafrost-coverage threshold**;
   report per-region AOA coverage; in-AOA only.
3. **Incumbent contrast — Form B, gated.** Against **Olefeldt et al. 2016** thermokarst
   landscape classes (the only Alaska-statewide comparable incumbent; wang2023 = lake
   occurrence >60°N, yang2024 = QTP, zhang2025 = stage). Reproject Olefeldt to grid, show
   **log-evidence spans a wide range within a single Olefeldt class** → mode is an
   **orthogonal axis** the categorical map doesn't resolve. **Positioning, NOT validation**
   (no statewide mode ground truth). **Gate on clean data acquisition; fall back to Form A
   (qualitative discussion paragraph) if alignment is hard. Form C (divergence-hotspot map,
   needs a contestable category→mode mapping) is ruled out.**

## Figure posture

- **Fig 1: lead clean with the susceptibility map** (user's call — not bias-first).
- **Merge the AOA into the map figure as a panel** so the product is never shown without
  its reliability mask.
- `output/aucpr_vs_blocksize.png` is publication-ready as-is (floor → logistic → XGBoost
  laddered with across-fold error bars).
- Full figure lineup to be *derived from the chain above* in the planning session, not
  pre-fixed.

## Deferred / noted (not blocking)

- **Methods-coherence note**: L6a family ranking = OOF fold-refit SHAP on training points;
  L7 dominance map = all-data model SHAP on the grid. Acceptable, but apply the grouped-family
  definitions **identically** across both so it can't be called inconsistent. (User: noted,
  not urgent.)
- Static/undated limitation (no thaw timing/rate — labels undated, `ImageryDates` dropped) →
  limitations section; deliverable is inherently static susceptibility.

## Key stamps (from FINDINGS.md)

70 features × 19,288 rows; prevalence 93.21/6.79 (positive = Non-abrupt, floor 0.0574);
model `models/model.json`; CV = albers_grid 10 km, buffer 0.0, 5×5 nested, seed 42;
grid 3229×2087, 2,849,807 valid cells (42.3% of grid), 2,773,804 in-AOA.
