# Manuscript outline

Section-by-section outline for the map-led *Earth's Future* draft, from the
2026-07-20 outlining session. Sits downstream of `STRATEGY.md` (argument spine +
committed framing) and `figures/FIGURES.md` (figure spec). This file is the
**prose skeleton**: section skeleton, per-section beats, and each beat's L-chain
and figure mapping. Numbers are stamped from `STRATEGY.md` / `diagnostics/FINDINGS.md`.

`main.tex` is authored by the lead author; this is the reference the drafting works from.

## Cross-cutting principles (apply everywhere)

- **Results follow figure order.** The order of beats in §4–§5 tracks the locked
  figure sequence, not the L1→L7 chain. The figure order is deliberately
  product-first (setup → map → credibility → drivers → downstream).
- **Results stay concise, factual, focused.** The epistemic arguing (predisposition≠occurrence,
  proxy-vs-mechanism defense, mechanistic interpretation) happens in the Discussion, not Results.
- **Language discipline:** "features more consistent with abrupt thaw" / feature-consistency —
  never "% susceptible", "% will thaw abruptly", or probability. The index is prior-free
  **log-evidence** (`>0` favors abrupt), not a calibrated probability, not a class.
- **Descriptive, not causal:** indicators *distinguish* abrupt from non-abrupt thaw — they do
  not cause it or result from it. Reserve mechanistic language only where a physical prior backs it.
- **Class encoding (fixed):** `0 = Abrupt` (~93%), `1 = Non-abrupt` (~7%); metric "positive" = Non-abrupt.

## Section skeleton (locked decisions)

Standard AGU, **separate Results/Discussion** (skeleton A). Data and Methods separate.
SHAP interpretation is a Result (§4.3); L7 downstream (ecoregion, Olefeldt) + limitations
are Discussion. Results is **one section with subsections** (not flat sections).

1. Introduction · 2. Data · 3. Methods · 4. Results · 5. Discussion · 6. Conclusions

**Anti-burial of the headline map:** the map is a dedicated Results subsection (§4.1),
reinforced by amplifiers *outside* the body flow — graphical abstract = the map;
first Key Point = the headline; PLS leads with the map; result-forward Intro closer;
full-width Fig 3; Conclusions callback.

---

## §1 Introduction *(L1)*

Stakes-first funnel, then concept→gap paired, then the two priming paragraphs, then bookend.

- **P1 — dual stakes.** Two parallel hooks: (i) permafrost carbon feedback and (ii)
  infrastructure damage. Abrupt thaw punches above its areal share on *both* — the reason
  mode matters.
- **P2–P3 — concept → gap (elegantly paired).** The *mode vs stage vs occurrence* distinction
  (mode = which pathway; the unmapped axis). Then the incumbent gap: existing products predict
  occurrence (wang2023 lakes >60°N), stage (zhang2025), or give categorical thermokarst classes
  (Olefeldt 2016) — **none predict mode, statewide, continuously.**
- **P4 — priming ¶1 (earning trust from biased data).** Problem→known-resolution, persuasive
  stance: (1) biased presence sampling (lake/road) inflates naive skill → spatial-block CV +
  leave-region-out; (3) rare positive class makes accuracy/AUC-ROC misleading → AUC-PR vs the
  prevalence floor. Grounds the work in data-driven susceptibility mapping (landslide analog) to
  inherit that literature's treatment of these exact problems. Every sentence sets up a Methods payoff.
- **P5 — priming ¶2 (what the product is + where valid).** (2) sample ≠ landscape prevalence →
  calibrated probability unrecoverable → prior-free log-evidence index; (4) extrapolation beyond
  training feature space is untrustworthy → applicability domain (AOA). Conceptual, no equations/numbers.
- **P6 — this study + "here we show."** Result-forward bookend into §4.1: ~a quarter of the
  in-AOA permafrost domain has features more consistent with abrupt than non-abrupt thaw.

## §2 Data

- **2.1 Labeled thaw points (Thaw Database).** Source; thaw-type → **Class `0=Abrupt` 93.21% /
  `1=Non-abrupt` 6.79%**; *n* = 19,288; the *negative* definition of non-abrupt (bundles diffuse
  gradual thaw + GTN-P/CALM monitoring, ~18% of the minority). **Sampling provenance (lake/road
  bias) planted here** — load-bearing for L3/L4c, so named at first contact with the data.
- **2.2 Geospatial features (70).** Grouped by provenance: remote sensing / community data
  products / climate reanalysis; by-family summary in text, **full provenance table → Appendix.**
  NaN structure surfaced (SoilGrids ~16% train-missing; MODIS fire-QA gap >70°N).
  Coordinates carried as **metadata only, quarantined from `X`**.
- **2.3 Prediction datacube.** Statewide grid 3229×2087; 2,849,807 valid cells (42.3%); Obu
  permafrost mask; 2,589,808 in-AOA.

## §3 Methods *(grouped to mirror the paper's arc)*

*Train/evaluate block:*
- **3.1 Feature-table construction + class encoding** (`Class = where(ThawType=='Abrupt',0,1)`).
- **3.2 XGBoost** — justified by native NaN handling (SoilGrids/MODIS gaps) + SHAP interpretability,
  **explicitly not** by beating logistic.
- **3.3 Spatial-block CV** — `albers_grid`, 10 km blocks, buffer 0.0, 5×5 nested, seed 42; motivated
  by 62%-within-1 km geographic interleaving. Matched-fold logistic baseline.
- **3.4 Metrics** — AUC-PR vs 0.0574 floor; AUC-ROC noted **once**, minority-insensitive;
  leave-region-out protocol.

*Product block:*
- **3.5 Log-evidence index** — `logit(P_model(abrupt|x)) − logit(π_sample)`; rationale sample≠landscape
  prevalence (payoff of priming ¶2). Kept in narrative order, not elevated.
- **3.6 AOA** — rank-CDF SHAP-weighted dissimilarity index; feature-space envelope threshold
  DI = 0.27 (99.9th pct of CV training DI, not a skill limit — AUC-ROC ~0.97–0.99 flat with DI);
  Spearman(DI, |residual|) = 0.489.
- **3.7 Train/serve parity gate** — *method here*; the *result* lands in §4.2.

*Interpretation block:*
- **3.8 Grouped SHAP** — OOF fold-refit TreeSHAP for the ranking; family-construction **procedure**
  (feature-space Spearman clustering, `1−|Spearman|`, complete linkage, gap-cut 0.449, categorical
  collapse); all-data model SHAP for the dominance map; **family definitions applied identically
  across both** (coherence note). Note the grouping is from *feature correlation*, not SHAP —
  surfaced to preempt circularity.

## §4 Results *(concise, factual; figure-ordered)*

### 4.1 The abrupt-thaw susceptibility map *(L5 + AOA/L4b — Fig 3, dedicated subsection)*
1. **Map + headline** — 26.6% of the in-AOA permafrost domain has features more consistent with
   abrupt than non-abrupt thaw (688,804 / 2,589,808 cells, log-evidence > 0; + km² once cell size
   confirmed). Anchored to log-evidence = 0 only.
2. **Immediate guardrail** — brief predisposition≠occurrence note (1–2 sentences; full argument → §5.1).
3. **Coherence counterweight** — median in-AOA log-evidence = −2.46: the *typical* cell favors
   non-abrupt; abrupt is the minority mode.
4. **Spatial pattern, descriptive only** — where abrupt-favoring concentrates (ice-rich lowlands,
   NW Alaska); mechanism deferred to §4.3/§5.2.
5. **AOA panel** — product shown with its reliability mask; AOA flags 9.1% of the valid domain.

### 4.2 Model evaluation *(L2/L3/L4c — Figs 4–5)*
1. **Performance + not-model-specific** *(Fig 5a)* — spatial-block-CV AUC-PR ≈ 0.85 (10 km 0.852),
   ≈15× the 0.0574 floor, across-partition σ 0.01–0.03 (20 reshuffles); laddered floor → logistic (0.776) → XGBoost (0.852);
   **margin +0.076 ± 0.019 framed as robustness, not superiority.** AUC-ROC ~0.98–0.99 once, minority-insensitive.
2. **Generalization / not-a-location-proxy** *(Fig 5b)* — leave-region-out decays gracefully to
   AUC-PR 0.54 @ ~251 km (~9× floor).
3. **Artifact controls** (a few sentences; details → Supplement) — shuffle → chance, dummies at floor,
   only 4 contradictory-label pairs: real, not an artifact.
4. **Representativeness** *(Fig 4)* — training points flatter/wetter/lower-drainage than the grid
   (Slope 0.74° vs 3.92°; HND 1 m vs 17 m; Open Water 0.43 vs 0.04). Closes the loop to the §4.1 AOA:
   this gap is *why* reliability is a separate layer.

### 4.3 Drivers of the susceptibility signal *(L6 — Figs 6–9)*
1. **Emergent feature families (opens the section)** *(Fig 6)* — 70 features collapse into 22
   coherent families by feature-space correlation; framed as *establishing the analysis unit before
   attributing* (inoculates against circularity). Dendrogram → Supplement.
2. **Ranking, honest multi-factor** *(Fig 6)* — single-panel magnitude; family bars colored by
   thematic domain + member counts + a **domain-aggregate inset** giving the correct feature-level
   totals. Families: Alpine relief 23%, Annual/dry-season temperature 16%, Land Cover 12%, Thermal
   continentality 9%… At the domain level **Relief ≈ Temperature co-lead (~25% each)**, then snow /
   land cover (14% each), precipitation (9%), seasonality (7%), soil (5%) — no single control.
   Cross-family climate subtotals are invalid (families overlap); the inset preempts the bad sum.
3. **Direction** — folded into Fig 7 (the standalone violin panel is dropped): abrupt-orientation
   stated, but nearly all families lean abrupt only because the sample is 94% abrupt (prevalence, not
   a per-feature finding); the prior-free reading is the log-evidence index + where SHAP crosses zero.
4. **Mechanism / dependence shapes** *(Fig 7)* — 3×3 own-SHAP dependence for the top-9 individual
   indicators (4 relief / 2 snow / 3 climate); *shapes reported as fact* (slope's flat-and-steep two
   regimes; temperature's warm-edge cliff), mechanistic reading reserved for §5.2.
5. **Land Cover mechanism** *(Fig 8)* — per-class decomposition of the Land Cover family SHAP (Open
   Water → abrupt; Sedge/Herbaceous → non-abrupt); the categorical analog of the dependence shapes,
   and the one family that discriminates in both directions.
6. **Spatial variation of drivers** *(Fig 9, dominance map)* — per-cell dominant **domain** across
   in-AOA cells (all-data model, AOA-restricted, hue-only + area-fraction inset); drivers vary
   regionally (evidence for the §5.2 proxy rebuttal).

## §5 Discussion *(paragraph-level outline from the 2026-08-13 grilling session)*

**Flow (locked):** read-the-surface opener → interpret drivers (4 ¶, ending on the proxy
defense) → the proxy defense *hinges* into limitations → landscape pattern → Olefeldt
positioning → implications → outlook. Limitations sit **after** the drivers, not first: the
opener carries the *reading-the-number* guidance, and the proxy defense sets up the honest
"here are the real limits" turn. ~10 paragraphs, ~2,925 words (budget per ¶ below); the lever
is per-¶ word count, not cutting content. Numbers stamped from the locked Results §4.

- **¶1 · How to read the surface** *(~150 w, no fig)* — the index is a reading instrument, not a
  forecast. Predisposition ≠ occurrence (features *consistent with* abrupt thaw, not area that
  *will* thaw). Log-evidence not probability; anchored only at 0 (likelihood ratio 1); prior-free.
  Counterweight: median −2.46 → typical cell favors non-abrupt ~10×, abrupt is the minority mode
  even at 24.7%. One clause: read only within the AoA.
- **¶2 · Drivers — mechanistic synthesis** *(~400 w, Figs 7–8)* — topic: the model recovers known
  abrupt-thaw pathways from features alone, unsupervised. **Relief / water-routing, two-regime
  slope:** flat / low HND / negative curvature / high flow accumulation → lowland *lake/wetland*
  thermokarst; steep slopes >10° → *hillslope* failure (slumps, detachments) — one family, two
  pathways. Anchor: 37% of points at slope 0° (open water); Open Water & Emergent Herbaceous
  Wetlands → abrupt, Sedge/Herbaceous strongest non-abrupt (Fig 8). **Thermal state:** cold MAAT →
  abrupt = cold, ice-rich terrain that collapses abruptly. Reserved mechanistic language only where
  a physical prior backs it (ground-ice/drainage; thermal state).
- **¶3 · Drivers — weaker signals as associations** *(~250 w, Fig 7)* — snow and precipitation
  contribute but lack a clean physical prior. High/declining SWE → abrupt; low isothermality, low
  precip trend → abrupt. State plainly these are correlational; a thermal-insulation reading is
  flagged, not asserted. The "distinguish, not cause" reminder lives here.
- **¶4 · Drivers — expected-but-absent** *(~350 w, Fig 6)* — drivers the field expects to lead
  (fire, yedoma, soil ice) rank below topography and temperature. Two explanations: (i) **data
  gaps** (MODIS fire-QA gap >70°N; SoilGrids ~16% NaN); (ii) **collinearity** — topography and
  temperature already carry the ice-rich/yedoma signal. Guardrail: low SHAP ≠ mechanistically
  unimportant; do not read this as "ice content doesn't matter." Tie-back: this is *why* we grouped
  SHAP by feature correlation before attributing.
- **¶5 · Drivers — proxy-vs-mechanism defense** *(~200 w, Fig 9)* — you might suspect the model just
  learned the Thaw Database's lake/road sampling bias; it didn't. (a) generalizes hundreds of km
  from any training point (LRO 0.54 @ 251 km, ~9× floor) — a pure sampling-proxy collapses
  off-sample; (b) dominant driver switches by region (Fig 9) — not one proxy everywhere. Honest
  concession: bias and mechanism aren't fully separable; we did not partial. **Hinge sentence** →
  into limitations. **Not partialling is a locked call.**
- **¶6 · Limitations** *(~250 w, no fig)* — hinge: real limits do bind. **Main — static/undated:** a
  snapshot of present predisposition, not when / where / how-fast; labels undated, `ImageryDates`
  dropped. Brief: no statewide mode ground truth — point-level evaluation only (one sentence).
  **Close on descriptive-not-causal** → pivot into the interpretive reading that follows.
  *(AoA/representativeness and soil-missingness deliberately dropped here — covered in Results /
  opener; not re-litigated.)*
- **¶7 · Landscape pattern** *(~400 w, Fig 10)* — **primary finding:** abrupt-favoring *dominates*
  the tundra, is *present* in the taiga, *rare* elsewhere. **Second finding:** the 24.7%
  de-averages sharply — Brooks Range 67%, Seward Peninsula 66%, Arctic Foothills 47%, down to ~1% in
  the southern mountains; the headline masks strong regional concentration. **Then tentatively:** the
  two-regime picture may manifest as two abrupt regions with different mechanisms (northern mountains
  hillslope vs coastal lowlands lake/wetland); within-region spread is wide → sub-regional variation
  too. One clause: in-AoA only, low-permafrost ecoregions dropped, per-region AoA reported.
- **¶8 · Positioning vs Olefeldt** *(~275 w, Fig 11)* — purpose up front: compared to Olefeldt not to
  validate our map, but to find the *new* information the mode axis adds. Quantitative: log-evidence
  spans nearly the full range within every Olefeldt class; class explains only 1–7% of variance
  (η²: Hillslope 7%, Lake 3%, Wetland 1%) → mode is a **nearly orthogonal axis**. Affirm Olefeldt's
  validity (landform type + occurrence potential — a different, valid question). **Complementary,
  never corrective.** Bridge: the two are combinable layers → into implications.
- **¶9 · Implications** *(~325 w, no fig)* — bookend the Intro's dual stakes: **carbon** (mode
  governs quantity/form/timing; a spatial prior for the feedback's largest uncertainty) +
  **infrastructure/communities** (screening layer for high-predisposition regions, incl. remote /
  Indigenous communities). **Multimodal risk assessment:** overlay mode with Olefeldt / other layers
  for richer products.
- **¶10 · Outlook** *(~325 w, no fig — ends the section)* — **reflexive targeting (bridge):** the map
  + AoA guides new field sampling / monitoring, including into out-of-AoA terrain — a way to break the
  lake/road bias that bounds the model. **More detailed / specific data:** better fire, soil,
  ground-ice layers (callback to ¶4) may promote currently-masked drivers. **Static → dynamic:** add
  timing/rate → trajectory; framed as one of this paper's biggest challenges and a **grand challenge
  for the field**. End here.

**Draft-time fix (needs author permission — `.tex` edit):** Results §4.1 points spatial-pattern
discussion to "§4.3," but the ecoregion pattern now lives in §5 (¶7); update that cross-ref.

## §6 Conclusions

Contribution (first statewide *continuous* map of thaw *mode*) · headline-in-context (predisposition,
AOA-bounded) · leading indicators (alpine relief, temperature) + regional driver variation ·
complementary to incumbents · outlook (targeting/monitoring value, carbon+infrastructure relevance,
static→dynamic).

## Back matter

- **Open Research statement** — Thaw Database; feature source datasets; Circumpolar Thermokarst (Olefeldt);
  Obu permafrost mask; code repo; archived model + datacube (DOIs filled later).
- **Appendix** — full 70-feature provenance/citation table.
- **Supplement** — artifact controls (shuffle/dummies/contradictory-label detail); full SHAP family
  membership + clustering detail; AOA DI detail (`aoa_di_map`, `aoa_threshold_decision`); calibration.

## Deferred / open

- **Deferred to post-writing (author's call):** Key Points (3), Abstract, Plain-language summary — written
  as distillations once the body exists.
- **Title:** kept as-is — *"Susceptibility and Spatial Signature of Abrupt Thaw Across Alaska's Permafrost Landscapes."*
- **TODO:** confirm datacube cell size to convert 26.6% → absolute km² for the abstract/§4.1.
- **TODO:** DOIs for the Open Research statement.
