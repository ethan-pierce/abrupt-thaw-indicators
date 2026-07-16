# Train/serve parity gate [T23]

Per-feature comparison of the training column (`features_clean.csv`, the exact model input) against the datacube pixel (`prediction_data.nc`), the surface the model is scored on. Matched-location method: each training point is nearest-indexed to its 1 km cell and compared there, so residual gaps are *construction*, not the lake-/road-biased sampling.

- Features: **70**, feature sets identical, same order: **True**.

- Obu in-domain cells (PerProb > 0): **2,849,807**; training points outside cube footprint (edge-clipped): **1**.


**Caveat:** the point path samples at the exact training coordinate; the cube samples at the cell centre (offset up to ~0.7 km). Fine native-scale features (terrain @10 m) therefore carry real sub-cell variance on top of any construction gap — read Spearman / scale-ratio, not bit-equality. Each matched-location flag is adjudicated by the **near-centre control** (parity recomputed on points sitting on their cell centre): if it converges to ~1, the gap is offset geometry, not construction.


## Verdict

✅ **PASS** — no genuine train/serve construction discrepancy.

- Matched-location: **60** clean, **10** offset-sensitive (no unit/transform slip — parity is limited by sub-cell geometry and rises toward the cell centre; see `ρ near` / `match near`), **0** genuine construction flag(s).

- Category-set subset: **2** non-background class(es) present in-domain but absent from training (silent reference-bucket absorption; see below — negligible area).


### What the offset flags reveal (not a bug — a *sampling* signal)

The offset-sensitive features are exactly the spatially singular / small-patch ones — `Upstream Area` and `Height Above Nearest Drainage` (near-zero on the drainage line, large one pixel away), the terrain derivatives (`Mean curvature (500 m)`, `Northness`/`Eastness`), and the small-patch land covers (`Open Water` train-active **0.43** vs serve **0.04**; `Dwarf Scrub`, `Sedge/Herbaceous`). Their construction is identical both sides (native / nearest sampling), but the divergence quantifies the documented **lake-/road-collection bias**: training points sit systematically in flatter, wetter, lower-drainage, more-open-water locations than the statewide 1 km grid the model scores (`Slope` train median 0.74° vs cube 3.92°; `HND` 1 m vs 17 m). This is a *representativeness* concern for the map (why calibration is suspect — SCOPE — and why the AOA layer T21 matters), not a construction defect this gate should block on.


## Continuous features — matched-location parity

`ratio` = median(|serve|)/median(|train|) (≈1 ⇒ same units/transform); `ρ` = Spearman at matched points; `ρ near` = Spearman on points sitting on their cell centre (the offset control); `nan_agree` = NaN-pattern agreement.


| feature | train med | serve med | ratio | ρ | ρ near | train NaN | serve NaN (pt/state) | nan_agree | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Elevation | 60.8 | 294.7 | 1.05 | 0.995 | 0.999 | 0.0% | 0.0%/0.1% | 1.000 | ✓ |
| Slope | 0.7368 | 3.917 | 2.84 | 0.708 | 0.787 | 0.0% | 0.0%/0.1% | 1.000 | ✓ |
| Mean curvature (500 m) | -1.916e-05 | -2.286e-06 | 0.941 | 0.396 | 0.594 | 0.0% | 0.0%/0.1% | 1.000 | ~ offset |
| Mean curvature (2 km) | -8.281e-07 | -5.876e-07 | 1 | 1.000 | 1.000 | 0.0% | 0.0%/0.1% | 1.000 | ✓ |
| Height Above Nearest Drainage | 1 | 17.3 | 3.4 | 0.685 | 0.807 | 0.1% | 0.3%/0.4% | 0.997 | ~ offset |
| Upstream Area | 0.02291 | 0.009956 | 0.431 | 0.038 | 0.169 | 0.1% | 0.3%/0.4% | 0.997 | ~ offset |
| Annual Mean Temperature | -104 | -63 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Mean Diurnal Range | 89 | 96 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Isothermality | 19 | 21 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Temperature Seasonality | 1.361e+04 | 1.3e+04 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Max Temperature of Warmest Month | 142 | 165 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Min Temperature of Coldest Month | -308 | -286 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Temperature Annual Range | 459 | 447 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Mean Temperature of Wettest Quarter | 79 | 91 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Mean Temperature of Driest Quarter | -149 | -132 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Mean Temperature of Warmest Quarter | 80 | 102 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Mean Temperature of Coldest Quarter | -260 | -231 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Annual Precipitation | 275 | 308 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Precipitation of Wettest Month | 59 | 63 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Precipitation of Driest Month | 9 | 10 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Precipitation Seasonality | 65 | 63 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Precipitation of Wettest Quarter | 134 | 154 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Precipitation of Driest Quarter | 36 | 38 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Precipitation of Warmest Quarter | 126 | 147 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Precipitation of Coldest Quarter | 41 | 46 | 1 | 1.000 | 1.000 | 0.6% | 0.7%/0.4% | 1.000 | ✓ |
| Flammability Index | 0 | 0.0002 | nan | 0.952 | 0.981 | 2.7% | 2.7%/1.0% | 0.996 | ✓ |
| Mean Annual SWE | 21.87 | 32.18 | 1 | 0.997 | 0.999 | 0.2% | 0.3%/0.4% | 0.998 | ✓ |
| Trend in SWE | -0.724 | -0.2796 | 1 | 0.991 | 0.999 | 0.2% | 0.3%/0.4% | 0.998 | ✓ |
| Trend in precipitation | -0.05929 | 2.654 | 1.01 | 0.999 | 1.000 | 0.2% | 0.3%/0.4% | 0.998 | ✓ |
| Trend in temperature | 0.08237 | 0.05672 | 1 | 0.997 | 0.998 | 0.2% | 0.3%/0.4% | 0.998 | ✓ |
| Time Since Last Fire | 24 | 24 | 1 | 0.773 | 0.957 | 12.9% | 12.9%/4.8% | 0.999 | ✓ |
| Burn Count | 0 | 0 | nan | 0.773 | 0.957 | 12.9% | 12.9%/4.8% | 0.999 | ✓ |
| Northness | 0 | 0 | nan | 0.418 | 0.594 | 0.0% | 0.0%/0.1% | 1.000 | ~ offset |
| Eastness | 0 | 0 | nan | 0.462 | 0.554 | 0.0% | 0.0%/0.1% | 1.000 | ~ offset |
| Soil Organic Carbon (0-30 cm) | 1908 | 1545 | 1 | 0.963 | 0.981 | 16.3% | 13.8%/5.2% | 0.894 | ✓ |
| Soil Organic Carbon (30-200 cm) | 731.8 | 599.4 | 1.01 | 0.920 | 0.947 | 16.3% | 13.8%/5.2% | 0.894 | ✓ |
| Nitrogen (0-30 cm) | 9108 | 6743 | 0.996 | 0.969 | 0.980 | 16.4% | 13.9%/5.1% | 0.890 | ✓ |
| Nitrogen (30-200 cm) | 4198 | 3145 | 0.999 | 0.953 | 0.974 | 16.4% | 13.9%/5.1% | 0.890 | ✓ |
| Bulk Density (0-30 cm) | 85.33 | 89.5 | 1 | 0.933 | 0.960 | 16.3% | 13.8%/5.2% | 0.894 | ✓ |
| Bulk Density (30-200 cm) | 117.7 | 121.1 | 1 | 0.917 | 0.950 | 16.3% | 13.8%/5.2% | 0.894 | ✓ |
| Sand (0-30 cm) | 349 | 327.5 | 0.998 | 0.975 | 0.986 | 16.4% | 13.9%/5.1% | 0.890 | ✓ |
| Sand (30-200 cm) | 417.7 | 395.4 | 1 | 0.981 | 0.989 | 16.4% | 13.9%/5.1% | 0.890 | ✓ |
| Clay (0-30 cm) | 182.7 | 150.8 | 1 | 0.940 | 0.964 | 16.4% | 13.9%/5.1% | 0.890 | ✓ |
| Clay (30-200 cm) | 168.6 | 154.5 | 1 | 0.956 | 0.975 | 16.4% | 13.9%/5.1% | 0.890 | ✓ |

## One-hot (categorical) features — matched-location parity

`match` = fraction of matched points with identical 0/1 value; `match near` = the same on points on their cell centre (offset control); `train mean` / `serve mean` = active fraction.


| feature | train mean | serve mean | match | match near | verdict |
|---|--:|--:|--:|--:|:--:|
| Yedoma | 0.2544 | 0.144 | 0.992 | 0.998 | ✓ |
| Land Cover (Dwarf Scrub) | 0.2442 | 0.226 | 0.799 | 0.868 | ~ offset |
| Land Cover (Sedge/Herbaceous) | 0.03251 | 0.08994 | 0.812 | 0.893 | ~ offset |
| Land Cover (Shrub/Scrub) | 0.1312 | 0.2607 | 0.877 | 0.907 | ~ offset |
| Land Cover (Developed, Low Intensity) | 0.002022 | 0.0005449 | 0.997 | 0.995 | ✓ |
| Land Cover (Open Water) | 0.4285 | 0.03765 | 0.657 | 0.753 | ~ offset |
| Land Cover (Evergreen Forest) | 0.02006 | 0.1347 | 0.980 | 0.983 | ✓ |
| Land Cover (Woody Wetlands) | 0.04552 | 0.04193 | 0.930 | 0.944 | ✓ |
| Land Cover (Barren Land (Rock/Sand/Clay)) | 0.006066 | 0.08261 | 0.987 | 0.991 | ✓ |
| Land Cover (Emergent Herbaceous Wetlands) | 0.07616 | 0.03556 | 0.881 | 0.872 | ~ offset |
| Land Cover (Deciduous Forest) | 0.01047 | 0.035 | 0.987 | 0.992 | ✓ |
| Land Cover (Grassland/Herbaceous) | 0.0009851 | 0.01747 | 0.999 | 0.999 | ✓ |
| Land Cover (Mixed Forest) | 0.001763 | 0.03095 | 0.996 | 0.997 | ✓ |
| Land Cover (Cultivated Crops) | 0.0001555 | 0.0001695 | 1.000 | 1.000 | ✓ |
| Land Cover (Developed, Open Space) | 5.185e-05 | 8.843e-05 | 1.000 | 1.000 | ✓ |
| Land Cover (Developed, High Intensity) | 5.185e-05 | 4.562e-06 | 1.000 | 1.000 | ✓ |
| Land Cover (Perennial Ice/Snow) | 5.185e-05 | 0.006173 | 1.000 | 1.000 | ✓ |
| Land Cover (Developed, Medium Intensity) | 0.0001555 | 3.72e-05 | 1.000 | 0.999 | ✓ |
| Land Cover (Pasture/Hay) | 5.185e-05 | 8.422e-06 | 1.000 | 1.000 | ✓ |
| Vegetation Mode (Graminoid tundra) | 0.3905 | 0.1443 | 0.930 | 0.973 | ✓ |
| Vegetation Mode (Shrub tundra) | 0.3485 | 0.2015 | 0.930 | 0.975 | ✓ |
| Vegetation Mode (White spruce) | 0.03432 | 0.1094 | 0.990 | 0.996 | ✓ |
| Vegetation Mode (Black spruce) | 0.02955 | 0.1062 | 0.990 | 0.995 | ✓ |
| Vegetation Mode (Deciduous forest) | 0.03468 | 0.269 | 0.998 | 0.999 | ✓ |
| Vegetation Mode (Wetland tundra) | 0.04402 | 0.01181 | 0.985 | 0.995 | ✓ |
| Vegetation Mode (Temperate rainforest) | 0.0001555 | 0.005601 | 1.000 | 1.000 | ✓ |

## Coverage: soil-NaN reproduction & fire QA gap

- **Soil** (10 composite cols): statewide serve NaN 5.1–5.2%, train NaN 16.3–16.4% (dry-run reported ~11.6% statewide). Soil is **native-sampled (250 m) on BOTH paths** (T35) — the T23 note's "soil 250 m→1 km reproject-averaging" concern is **stale**; there is no reproject-average, and the high matched ρ confirms identical construction.
- **Fire** (MODIS, T36): statewide serve NaN 4.8–4.8% — the documented >70°N QA coverage gap, reproduced on the serve side.

## Category-set subset check (silent reference-bucket absorption)

Raw NLCD / ALFRESCO classes sampled at in-domain cell centres. A non-background class present in-domain **without** a model one-hot column is folded silently into the dropped reference bucket.


### Land Cover

| code | class | in-domain cells | % domain | model column? |
|--:|---|--:|--:|:--:|
| 52 | Shrub/Scrub | 743,061 | 26.07% | ✓ |
| 51 | Dwarf Scrub | 644,187 | 22.60% | ✓ |
| 42 | Evergreen Forest | 383,808 | 13.47% | ✓ |
| 72 | Sedge/Herbaceous | 256,321 | 8.99% | ✓ |
| 31 | Barren Land (Rock/Sand/Clay) | 235,412 | 8.26% | ✓ |
| 90 | Woody Wetlands | 119,498 | 4.19% | ✓ |
| 11 | Open Water | 107,286 | 3.76% | ✓ |
| 95 | Emergent Herbaceous Wetlands | 101,332 | 3.56% | ✓ |
| 41 | Deciduous Forest | 99,751 | 3.50% | ✓ |
| 43 | Mixed Forest | 88,212 | 3.10% | ✓ |
| 71 | Grassland/Herbaceous | 49,799 | 1.75% | ✓ |
| 12 | Perennial Ice/Snow | 17,591 | 0.62% | ✓ |
| 22 | Developed, Low Intensity | 1,553 | 0.05% | ✓ |
| 74 | Moss | 1,097 | 0.04% | ❌ NO COLUMN |
| 82 | Cultivated Crops | 483 | 0.02% | ✓ |
| 21 | Developed, Open Space | 252 | 0.01% | ✓ |
| 23 | Developed, Medium Intensity | 106 | 0.00% | ✓ |
| 81 | Pasture/Hay | 24 | 0.00% | ✓ |
| 0 | code 0 | 21 | 0.00% | background (dropped) |
| 24 | Developed, High Intensity | 13 | 0.00% | ✓ |

**⚠️ 1 class(es) absorbed**, 1,097 in-domain cells (~1,097 km²): Moss

### Vegetation Mode

| code | class | in-domain cells | % domain | model column? |
|--:|---|--:|--:|:--:|
| 3 | Deciduous forest | 766,713 | 26.90% | ✓ |
| 4 | Shrub tundra | 574,247 | 20.15% | ✓ |
| 5 | Graminoid tundra | 411,155 | 14.43% | ✓ |
| 0 | code 0 | 404,251 | 14.19% | background (dropped) |
| 2 | White spruce | 311,816 | 10.94% | ✓ |
| 1 | Black spruce | 302,550 | 10.62% | ✓ |
| 6 | Wetland tundra | 33,653 | 1.18% | ✓ |
| 8 | Temperate rainforest | 15,962 | 0.56% | ✓ |
| 7 | Barren lichen moss | 18 | 0.00% | ❌ NO COLUMN |

**⚠️ 1 class(es) absorbed**, 18 in-domain cells (~18 km²): Barren lichen moss

---
_Generated by `diagnostics/train_serve_parity.py` (T23)._
