# PIPELINE

End-to-end provenance for the abrupt-thaw susceptibility pipeline: data sources,
the script→artifact DAG in execution order, and the final outputs. Describes the
**target** pipeline after the methods-cleanup rewrite (README to-do "Methods
cleanup", 2026-07-10); items that differ from the current code are marked *(new)*.
The per-run **manifest** (git SHA, data hashes, seeds, hyperparameters) is stamped
at run time beside `model.json` and is the reproducibility key for a specific run.

## 1. Data sources

### Labels
- **Alaska Permafrost Thaw Database v2.0.0** — `data/Alaska_Permafrost_Thaw_Database_v2.0.0.csv`
  (`webb2026-thawdb`, `REFERENCES.md`; 19,540 rows, 93.21%/6.79% abrupt/non-abrupt).

### Features — public datasets (Google Earth Engine)
| Feature(s) | Source | GEE id | Reducer @ scale |
| --- | --- | --- | --- |
| Land Cover | USGS **NLCD 2016** (AK) | `projects/ee-abrupt-thaw/assets/NLCD-2016` | mode @ 30 m |
| Elevation, Slope, Aspect | USGS **3DEP 10 m** DEM | `USGS/3DEP/10m` | mean @ 10 m |
| 19 bioclimatic variables | **WorldClim v1** BIO | `WORLDCLIM/V1/BIO` | mean @ 1 km |
| SOC, Nitrogen, Clay, Sand, Silt, Bulk Density (6 depths each) | **SoilGrids** (ISRIC, 250 m) | `projects/soilgrids-isric/{soc,nitrogen,clay,sand,silt,bdod}_mean` | mean @ 250 m |

### Features — custom uploaded assets (`projects/ee-abrupt-thaw/assets/…`)
| Feature | Asset | Upstream source | Reducer @ scale |
| --- | --- | --- | --- |
| Mean curvature (500 m, 2 km) | `AK-curvature-500m`, `AK-curvature-2k` | derived from DEM (confirm) | mean @ 100 m |
| Flammability Index | `ALFRESCO-historical-flammability` | UAF SNAP **ALFRESCO** | mean @ 1 km |
| Vegetation Mode | `ALFRESCO-historical-vegetation-mode` | UAF SNAP **ALFRESCO** | mode @ 1 km |
| Maximum Fire Temperature | `max-fire-temp` (band `T21`) | NASA **FIRMS** (MODIS ~4 µm brightness temp, K) | mean @ 1 km |
| Mean Annual SWE | `ee-mean-annual-swe` | **confirm upstream** | mean @ 1 km |
| Trend in SWE / precip / temp | `annual-swe-trend`, `annual-precip-trend`, `temp-trend` (band `scale`) | **confirm upstream** | mean @ 1 km |
| Projected summer/winter temp change, precip change | `summer-temperature-trend`, `winter-temperature-trend`, `annual-precipitation-trend` | **confirm upstream** | mean @ 1 km |

> **Confirm:** the SWE, trend, projected-climate, and curvature assets are custom
> uploads whose upstream product/DOI isn't in the code. Fill these in for the
> manuscript methods table.

### Masks (prediction domain / reliability)
- **Permafrost domain** — Obu et al. 2019 permafrost probability (PerProb 5.0),
  `data/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH/` (PANGAEA). *(new — G17)*
- **Categorical cross-check (optional)** — Brown et al. 1997 Circum-Arctic Permafrost
  and Ground-Ice map, `data/arctic-permafrost-map/` (NSIDC GGD318).

## 2. Execution order (script → artifact)

1. **`data/build_feature_table.py`** *(GEE; needs `ee.Authenticate()`, project `ee-abrupt-thaw`)*
   reads the v2.0.0 database + all §1 feature sources → **`data/features_dirty.csv`**.
2. **`data/clean_feature_table.py`** reads `features_dirty.csv` → **`data/features_clean.csv`**.
   *(new)* derives the `Fire Detected` indicator (A1); carries `Latitude`/`Longitude`
   as non-model columns (B6); dedup subset excludes coords (A3).
3. **`models/train_xgboost.py`** reads `features_clean.csv` → **`models/model.json`** +
   **run manifest** + evaluation figures. *(new)* nested spatial CV over an equal-area
   (EPSG:3338) km-grid block-size sweep, 1 km buffer (B4/B5); selects on pooled-OOF
   AUC-PR (C8); `scale_pos_weight=1` (C9); logistic + dummy baseline diagnostics (D12);
   persists CV config + seeds (B6).
4. **`data/build_prediction_data.py`** *(GEE)* reads `roi.geojson` + §1 sources →
   **`data/prediction_data.nc`** (statewide 4 km datacube, feature-name-driven).
5. **`models/predict.py`** reads `prediction_data.nc`, `model.json`, Obu PerProb →
   **continuous log-evidence susceptibility surface** + **AOA mask**. *(new)* emits the
   log-evidence index (E13), Obu-masked (G17), with the AOA reliability layer (G18);
   **no discrete classification** (G19).
6. **`models/shap_values.py`** reads `features_clean.csv`, `model.json`, CV config →
   **pooled out-of-fold SHAP** outputs (F14/F15).

*Archived:* the training-lands path (`build_prediction_data_traininglands.py`,
`predict_traininglands.py`, `*_traininglands.*`) — README to-do #8.

## 3. Final outputs

- **`models/model.json`** + run manifest (git SHA, `features_clean.csv` hash, CV
  config + seeds, Obu/Brown product versions, selected hyperparameters). *(H20.1)*
- **Performance**: AUC-PR (positive = Gradual) vs. block-size curve with across-fold
  spread + prevalence floor; AUC-ROC secondary; baseline diagnostics. *(no accuracy)*
- **Continuous log-evidence abrupt-thaw susceptibility surface** (statewide 4 km,
  masked by Obu permafrost domain) — the single headline map.
- **Area-of-Applicability / dissimilarity mask** — extrapolation-reliability layer.
- **Pooled out-of-fold SHAP** — importance ranking, beeswarm, dependence plots.
- **`diagnostics/` re-run** — updated `/verify-ml` suite + regenerated `FINDINGS.md`
  verifying the new invariants (coord quarantine, buffer, OOF SHAP, baselines). *(H20.2)*
