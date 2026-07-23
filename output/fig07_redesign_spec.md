# Figure 7 redesign spec — SHAP family importance (single panel)

**Status:** ready for sub-agent handoff. All decisions closed in the 2026-07-23 grilling session.
**Target script:** edit `output/fig07_shap_families.py`.
**Outputs:** `output/07_shap_families.{pdf,png}` (unchanged filenames).
**Do NOT touch:** `models/shap_groups.py`, family definitions, or the `mean|Σ member SHAP|`
metric — the grouping and metric are sound and stay. This is a presentation change only.
Do not edit any `.tex`.

---

## What changes (summary)

1. **Drop panel (b), the direction violins.** Fig 7 becomes a **single-panel** magnitude figure.
2. **Color the family bars by thematic domain** (was plain teal).
3. **Annotate each bar with its member count** (`n=<k>`).
4. **Add an inset bar strip** giving the honest domain-level aggregates.
5. Keep the 22 emergent families, importance-sorted, and the per-family `%` annotation.

## Why

The old figure listed 22 equal peer rows; seven are climate, so readers subtotal them into a
false "climate ≈ 44%" claim. That cross-family sum is illegitimate (families remain mutually
correlated, |ρ| up to 0.55). The per-row numbers are honest — the figure just needs to (a) show
the correct thematic structure and (b) stop inviting the bad sum. Panel (b) is dropped because
its only real justification — surfacing bidirectional families a signed bar would erase — applied
solely to **Land Cover**, which now has its own figure; every other important family is cleanly
one-directional and its direction+shape is shown per-feature in Fig 8. (See "Prevalence note".)

## The honest result being communicated (accept, don't minimize)

A **multi-factor** signal. Correct thematic aggregation shows **Relief and Temperature co-lead at
~25% each — neither dominates** — with snow, land cover, precipitation, seasonality, and soil all
carrying real weight. Single strongest *feature* = Slope; strongest *family* = Alpine relief;
strongest *domains* = Relief ≈ Temperature.

---

## Panel — magnitude (the only panel)

- Horizontal bars, **all 22 emergent families**, importance-sorted (most important on top).
  Data unchanged: read family labels/importance/per-point matrix from
  `output/shap_grouped_matrix.npz` (as now).
- **Bar fill = the family's domain color** (palette below).
- **Member-count annotation** `n=<k>` on/beside each bar (k = `n_members` from the family record),
  so redundancy is visible (e.g. *Precipitation amount* 5% is n=7; *Upstream Area* 3% is n=1).
- Keep the existing per-family `%`-of-summed-importance annotation.
- x-axis label unchanged: "Mean |Σ member SHAP| (margin)".
- Now a single tall panel; place the inset (below) in the lower-right whitespace left by the short
  tail bars.

### Family → domain color assignment (FINAL — 7 domains appear at family level)

| domain | families |
|---|---|
| Relief / topography | Alpine relief; Mean curvature (500 m); Mean curvature (2 km); Upstream Area; Eastness; Northness |
| Temperature | Annual / dry-season temperature; Thermal continentality; Summer warmth; Trend in temperature |
| Precipitation | Precipitation amount; Trend in precipitation |
| Seasonality | Isothermality / precip seasonality |
| Soil | Sand fraction; Clay fraction; Bulk density; Soil organic / fertility |
| Land cover / veg / fire | Land Cover; Vegetation Mode; Flammability Index; Fire history |
| Other / ground ice | Yedoma |

(Families are colored by their **dominant** domain. Note two families carry a snow member that is
counted under "Snow" in the inset — see Snow note — this is expected, not an error.)

## Inset — domain-aggregate bar strip (the anti-subtotal device)

A small horizontal bar strip, placed in the panel's lower-right whitespace, titled:
**"Thematic aggregate (feature-level). Families share information and cannot be summed — these are
the correct domain totals."**

8 domain bars, sorted descending, each in its domain color, labeled with share:

| domain | share |
|---|---|
| Relief / topography | 24.7% |
| Temperature | 24.6% |
| Snow (SWE) | 14.3% |
| Land cover / veg / fire | 14.3% |
| Precipitation | 9.1% |
| Seasonality | 6.9% |
| Soil | 5.1% |
| Yedoma / ground ice | 1.1% |

**Compute these live** (do not hard-code) from the per-feature SHAP in
`output/shap_mechanism_cache.npz`: for each domain, `mean over points of |Σ over its features of
SHAP|`; share = domain aggregate / sum of the 8 domain aggregates. The two caches are the same OOF
SHAP, so this is consistent with the family bars. Print the computed table on run; it should match
the above (±0.1).

### Feature → domain map for the inset (FINAL — assign categoricals FIRST)

**Single source of truth:** put this feature→domain map AND the domain color palette in
`output/shap_domains.py` and import it here — Fig 9 (the dominance map) imports the SAME module, so
the two figures cannot drift. Create the module if it does not exist yet.

Assign every one-hot `Land Cover (*)` and `Vegetation Mode (*)` to **Land cover/veg/fire** BEFORE
any keyword matching (otherwise "Land Cover (Barren Land (Rock/Sand/Clay))" false-matches Soil).

- **Snow** (2): Mean Annual SWE, Trend in SWE
- **Seasonality** (2): Isothermality, Precipitation Seasonality
- **Temperature** (11): Annual Mean Temperature, Mean Temperature of Driest Quarter, Mean Diurnal
  Range, Temperature Seasonality, Min Temperature of Coldest Month, Temperature Annual Range, Mean
  Temperature of Coldest Quarter, Max Temperature of Warmest Month, Mean Temperature of Wettest
  Quarter, Mean Temperature of Warmest Quarter, Trend in temperature
  *(Temperature Seasonality stays here, not Seasonality — it lives in the Thermal-continentality
  family; only the two features of the named "Isothermality / precip seasonality" family go to
  Seasonality.)*
- **Precipitation** (8): Annual Precipitation, Precipitation of Wettest Month, Precipitation of
  Driest Month, Precipitation of Wettest Quarter, Precipitation of Driest Quarter, Precipitation of
  Warmest Quarter, Precipitation of Coldest Quarter, Trend in precipitation
- **Relief** (8): Slope, Elevation, Height Above Nearest Drainage, Mean curvature (500 m), Mean
  curvature (2 km), Upstream Area, Eastness, Northness
- **Soil** (10): Sand (0-30 cm), Sand (30-200 cm), Clay (0-30 cm), Clay (30-200 cm), Bulk Density
  (0-30 cm), Bulk Density (30-200 cm), Soil Organic Carbon (0-30 cm), Soil Organic Carbon
  (30-200 cm), Nitrogen (0-30 cm), Nitrogen (30-200 cm)
- **Land cover / veg / fire** (28): all `Land Cover (*)` (18) + all `Vegetation Mode (*)` (7) +
  Flammability Index + Time Since Last Fire + Burn Count
- **Other / ground ice** (1): Yedoma

(Total 70. Snow and Seasonality are feature-level domains only — they have no family bar.)

### Snow note (expected wrinkle, state in caption)

Snow (14.3%) is two features — Mean Annual SWE and Trend in SWE — that live *inside* the Alpine
relief and Thermal continentality **families**, so Snow appears in the inset but has no family bar.
Footnote it: "Snow (SWE) features sit within the relief and thermal families; broken out here to
show their thematic weight."

## Prevalence note (caption sentence — prevents over-reading direction)

The sample is 94% Abrupt, so nearly all families lean warm (toward Abrupt) *relative to the
abrupt-heavy sample average* — this is prevalence, not a claim that every indicator marks abrupt
thaw. The prior-free reading is the log-evidence index and the Fig 8 dependence shapes (where SHAP
crosses zero), not the absolute sign. (This is why direction moved to Fig 8 and why panel (b) was
dropped.)

## Framing discipline (locked)

Attribution fact, not mechanism or necessity. "The model's evidence leans on …", never "… causes
…". Necessity (drop-the-group refit) is out of scope (deferred) — don't imply it.

## Color / mechanics

- Define a **categorical, CVD-safe domain palette** (8 colors incl. Snow) in `figstyle` for reuse —
  **Fig 9's dominance map MUST use the same domain colors.** It must be visually distinct from
  `figstyle.ABRUPT`/`NON_ABRUPT` (warm/cool sign axis) and from Fig 6's green/purple.
- Use `figstyle` (`use()`, `save`, AGU width, panel label). Single panel; size taller-than-wide to
  fit 22 rows; the old two-panel width can shrink toward one column.
- Print a run summary: family order, computed domain-aggregate table, palette used.

## Coherence / cross-references (put in caption)

- Fig 8 (per-feature dependence) nests into these families — e.g. Slope is the lead member of
  Alpine relief.
- Land Cover per-class direction → its own figure (the sole bidirectional family).
- Fig 9 (dominance map) uses the SAME 22 families and the SAME domain colors.
