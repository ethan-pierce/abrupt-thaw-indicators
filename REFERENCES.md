# REFERENCES

Durable record of sources actually opened and read, kept as the project positions
itself against the field. An entry means the source was opened — presence here
is the verification mark, not a status tag. Each entry: citation, what it shows
in its own terms, what it bears on in this project.

## webb2026-thawdb

**Citation:** Webb, H., Pierce, E., Abbott, B.W., Bowden, W.B., Chen, Y., Chen, Y.,
Douglas, T.A., Eklof, J.F., Euskirchen, E.S., Langer, M., Myers-Smith, I.H.,
Overeem, I., Strauss, J., Walter Anthony, K., Wang, K., Whitley, M.A., Turetsky,
M.R. (2026). A comprehensive database of thawing permafrost locations across
Alaska: version 2.0.0. *Earth System Science Data*, 18, 3147–3164.
https://doi.org/10.5194/essd-18-3147-2026

**What it shows:** Data synthesis (measurement compilation), full text opened.
Integrates 44 sources (field observations + remote sensing), 1950–present, into
19,540 permafrost thaw locations across all Alaska ecoregions: 18,213 abrupt
(10,625 of those thermokarst-lake features) vs. 1,327 non-abrupt — a 93.2%/6.8%
split, consistent with this repo's stated ~93%/~7% class balance. States
sampling is uneven (denser along road systems) and that lake dominance "could
skew model outputs toward aquatic thaw processes while underrepresenting
terrestrial forms." Finds only 65% of ice-dependent abrupt-thaw locations fall
within areas mapped as high/moderate ground ice by existing products, and
concludes those products "were not designed to capture fine-scale heterogeneity
in ground ice conditions."

**What it bears on:** This is the direct source of this repo's Thaw Database
(confirmed directly by the user, second author on this paper) — resolves
`SCOPE.md`'s previously-flagged "likely data lineage" open item from a guess to
a fact; a data-version/re-run reconciliation remains a separate, deferred issue.
Also grounds two SCOPE.md claims with numbers: the terrestrial/aquatic sampling-bias
caveat, and the "categorical ground-ice maps are coarse" half of objective 4's
positioning claim.

## zhang2025-thawstage

**Citation:** Zhang, C., Douglas, T.A., Brodylo, D., Jorgenson, M.T., Bosche, L.V.
(2025). Mapping permafrost thaw stages in interior Alaska. *Remote Sensing of
Environment*, 329, 114941. https://doi.org/10.1016/j.rse.2025.114941

**Access caveat:** Read via indexed abstract/excerpts only; full text paywalled
(ScienceDirect 403, SSRN preprint 403).

**What it shows:** Classification study. Combines repeat airborne LiDAR with
WorldView-2/Sentinel-2/Landsat time series and terrain data over a 2500 km²
fire-influenced interior-Alaska landscape; classifies four post-fire thaw
*stages* (old, lateral, vertical-shallow, vertical-deep thaw) at 79% overall
accuracy; finds the thaw pattern is fire-controlled and locally modified by
other drivers.

**What it bears on:** Objective 4 — the closest same-region ML precedent, but
the target variable is thaw *stage/severity* over a known post-fire timeline,
not abrupt-vs-gradual thaw *mode*; the modality is repeat-LiDAR change detection
over a 2500 km² footprint, not a static statewide point feature-stack + SHAP.
The contrast is target and scale, not "ML vs. no ML."

## yang2023-rts-cnn

**Citation:** Yang, Y., Rogers, B.M., Fiske, G., Watts, J., Potter, S.,
Windholz, T., Mullen, A., Nitze, I., Natali, S.M. (2023). Mapping retrogressive
thaw slumps using deep neural networks. *Remote Sensing of Environment*, 288,
113495. https://doi.org/10.1016/j.rse.2023.113495

**Access caveat:** Read via a Woodwell/Permafrost Pathways research-communications
summary (opened directly) plus indexed abstract excerpts; full text not
independently opened.

**What it shows:** CNN trained on 965 labeled retrogressive-thaw-slump (RTS)
features (509 Yamal/Gydan, 456 across six other pan-Arctic regions incl.
Canada) to detect discrete RTS footprints from high-resolution imagery +
topography, for annual-update monitoring of slump size/distribution. Does not
address gradual thaw or classify thaw mode; does not reference Olefeldt-style
categorical susceptibility maps.

**What it bears on:** Objective 4 — contrasts on target and output type: object
detection of one discrete abrupt-thaw landform (RTS), not point-level
classification of abrupt-vs-gradual mode across mechanisms, and discrete
detections rather than a continuous probability surface. A 2025 follow-up
(DARTS, circum-Arctic RTS database, *Scientific Data*) extends this line of
work further but was only seen in search results, not opened — left as an
unopened lead.

## webb2025-definitions

**Citation:** Webb, H., Fuchs, M., Abbott, B.W., et al. (2025). A review of
abrupt permafrost thaw: definitions, usage, and a proposed conceptual
framework. *Current Climate Change Reports*, 11(1), article 7.
https://doi.org/10.1007/s40641-025-00204-3

**What it shows:** Synthesis/review of terminology across 226 studies. Finds
"abrupt thaw" used inconsistently, spanning three senses (thermokarst/thermal-erosion
feature formation, rapid rate, or both). Proposes a definition requiring both:
(1) faster-than-typical top-down permafrost degradation, initiating within
~30 years, via either ice-rich internal feedbacks or external drivers (fire,
hydrologic change, gas buildup) even in ice-poor ground, and (2) substantial,
persistent alteration of ecosystem structure/function — vs. gradual thaw as
slow, linear active-layer thickening over years to decades. Proposes a
rule-based decision tree (timescale × ecosystem-change magnitude × ice-content
dependency), explicitly categorical, not continuous/probabilistic. States
fine-scale ground-ice mapping from satellite alone remains unresolved, and
names AI/data-driven multi-source methods as "a promising path" for abrupt-thaw
mapping — not yet realized in the literature it reviews.

**What it bears on:** Sharpens the glossary's "Abrupt thaw" / "Gradual thaw"
one-liners with a reviewed, citable definition (routed to `/ideate` as an
option, not applied here). Also grounds `SCOPE.md`'s Key background incumbent-view
claim that current mapping is rule-based/categorical by design — and directly
supports objective 4's framing: this review explicitly names the AI/data-driven
gap this project fills, as of a source current through its 2025 review window.

## yang2024-qtp-interpretable

**Citation:** Yang, Y., Wang, J., Mao, X., Lu, W., Wang, R., Zheng, H. (2024).
Susceptibility modeling and potential risk analysis of thermokarst hazard in
Qinghai–Tibet Plateau permafrost landscapes using a new interpretable ensemble
learning method. *Atmosphere*, 15(7), 788. https://doi.org/10.3390/atmos15070788

**Access caveat:** Read via indexed abstract/excerpts only (MDPI 403);
bibliographic details confirmed via CrossRef.

**What it shows:** Stacking ensemble (random forest, extremely randomized
trees, XGBoost, CatBoost) predicting thermokarst-hazard occurrence/susceptibility
across Qinghai–Tibet Plateau permafrost, interpreted with SHAP, LIME, and ALE
together; ~91% of known hazard points fall within predicted high/very-high
susceptibility zones, covering ~20% of the QTP permafrost extent.

**What it bears on:** Objective 4 — **counter-evidence found via a deliberate
counter-source hunt.** Continuous, ML-based, SHAP-interpreted susceptibility
mapping for permafrost hazards is already established practice, at least in
the QTP literature. This narrows the honest novelty claim: it is not
"continuous/ML vs. categorical" in general (already done elsewhere), but that
this project classifies abrupt-vs-gradual thaw *mode* across thaw mechanisms
together (not one hazard type's occurrence probability), for Alaska/pan-Arctic
tundra permafrost specifically.

## wang2023-arctic-tls

**Citation:** Wang, R., Guo, L., Yang, Y., Zheng, H., Jia, H., Diao, B., Li, H.,
Liu, J. (2023). Thermokarst lake susceptibility assessment using machine
learning models in permafrost landscapes of the Arctic. *Science of the Total
Environment*, 900, 165709. https://doi.org/10.1016/j.scitotenv.2023.165709

**Access caveat:** Read via indexed abstract/excerpts only (ScienceDirect 403);
bibliographic details confirmed via CrossRef.

**What it shows:** Model comparison (random forest performed best) producing
the first Arctic-wide (poleward of 60°N) thermokarst-*lake* susceptibility map;
~10.4% of the region (1.8M km²) rated high/very-high susceptibility; slope is
the dominant conditioning factor.

**What it bears on:** Objective 4 — same counter-evidence as
[[yang2024-qtp-interpretable]], but Arctic-wide rather than QTP, which
strengthens the correction: continuous ML susceptibility mapping in this
project's own biome/latitude band is not new. Still lake-occurrence-specific
(excludes hillslope/wetland thaw forms this project's Thaw Database also
covers) and does not classify abrupt-vs-gradual mode.
