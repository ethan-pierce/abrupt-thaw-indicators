# abrupt-thaw-indicators

Code for "Susceptibility to Abrupt Thaw across Alaska's Permafrost Landscapes" (Earth's Future, in review). A gradient-boosted (XGBoost) classifier learns which geospatial indicators separate abrupt from non-abrupt permafrost thaw at 19,288 expert-labeled sites in the Alaska Permafrost Thaw Database. The model is evaluated under spatial cross-validation, interpreted with grouped SHAP values, and applied to a statewide 1 km feature grid. The main product is a prior-free log-evidence index of thaw mode across Alaska's permafrost domain, paired with an area-of-applicability layer that flags where the grid lies outside the training data.

## Citation

- **Paper:** Pierce, E., Overeem, I., Webb, H., Rozmiarek, K., & Turetsky, M. (in review). Susceptibility to Abrupt Thaw across Alaska's Permafrost Landscapes. *Earth's Future*.
- **Code:** Pierce, E. (2026). abrupt-thaw-indicators v1.0. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
- **Data:** Pierce, E., Overeem, I., Webb, H., Rozmiarek, K., & Turetsky, M. (2026). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

The data record holds the susceptibility map (`log_evidence`, `DI`, `inside_aoa`), the prediction datacube, the training feature table, and a data dictionary.

## Layout

```
data/          feature extraction (GEE + local rasters), cleaning, datacube build, Zenodo export
models/        spatial CV, training, SHAP, prediction, area of applicability; model.json and run config
diagnostics/   analyses behind specific numbers in the paper (repeated CV, extrapolation range,
               buffer sweep, univariate separation, AoA threshold calibration)
output/        figure scripts, figure style, and rendered figures
manuscript/    LaTeX source (excluded from release archives)
settings.py    paths, metadata column names, and the Earth Engine project
```

Class encoding throughout: `0 = Abrupt`, `1 = Non-abrupt`.

## Setup

Requires Python 3.13 and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/ethan-pierce/abrupt-thaw-indicators.git
cd abrupt-thaw-indicators
poetry install
```

Steps marked **GEE** below need Google Earth Engine. Scripts call `ee.Initialize(project=settings.EE_PROJECT)` and fall back to `ee.Authenticate()` when there are no cached credentials. Set `EE_PROJECT` in `settings.py` to your own Cloud project, or run `poetry run earthengine authenticate` once.

## Data inputs

Third-party data are not redistributed. Earth Engine sources are read in place; the rest are expected at the paths below (all git-ignored).

| Product | Access | Local path | Used by |
| --- | --- | --- | --- |
| Alaska Permafrost Thaw Database v2.0.0 ([10.5281/zenodo.17494851](https://doi.org/10.5281/zenodo.17494851)), CC0 | included | `data/Alaska_Permafrost_Thaw_Database_v2.0.0.csv` | feature table |
| USGS 3DEP 10 m DEM ([ScienceBase](https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5)) | GEE `USGS/3DEP/10m` | — | features |
| MERIT Hydro v1.0.1 ([project page](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/)) | GEE `MERIT/Hydro/v1_0_1` | — | features |
| WorldClim v1.4 bioclimatic variables ([worldclim.org](https://www.worldclim.org/data/v1.4/worldclim14.html)) | GEE `WORLDCLIM/V1/BIO` | — | features |
| SoilGrids 2.0 ([10.5194/soil-7-217-2021](https://doi.org/10.5194/soil-7-217-2021)) | GEE `projects/soilgrids-isric/*_mean` | — | features |
| Daymet V4 ([10.3334/ORNLDAAC/1840](https://doi.org/10.3334/ORNLDAAC/1840)) | GEE `NASA/ORNL/DAYMET_V4`, reduced by `build_daymet_rasters.py` | `data/daymet/daymet_v4_reductions_1km_3338.tif` | features |
| MODIS MCD64A1 v6.1 ([10.5067/MODIS/MCD64A1.061](https://doi.org/10.5067/MODIS/MCD64A1.061)) | GEE `MODIS/061/MCD64A1`, reduced by `build_modis_fire_rasters.py` | `data/modis_fire/mcd64a1_fire_history_500m_3338.tif` | features |
| ALFRESCO flammability and vegetation mode ([SNAP](https://data.snap.uaf.edu/data/IEM/Outputs/ALF/Gen_1a/)) | download with `fetch_alfresco.py` | `data/alfresco/` | features |
| NLCD 2016 Alaska ([10.5066/P96HHBIE](https://doi.org/10.5066/P96HHBIE)) | manual download | `data/NLCD2016/NLCD_2016_Land_Cover_AK_20200724.img` | features |
| IRYP v2 yedoma ([10.1594/PANGAEA.940078](https://doi.org/10.1594/PANGAEA.940078)) | manual download | `data/IRYP_v2_yedoma_confidence_Shapefile/IRYP_v2_yedoma_confidence.shp` | features |
| Obu et al. permafrost probability ([10.1594/PANGAEA.888600](https://doi.org/10.1594/PANGAEA.888600)) | manual download | `data/Obu2019/UiO_PEX_PERPROB_5.0_20181128_2000_2016_NH.tif` | domain mask, Fig 2 |
| Circumpolar Thermokarst Landscapes ([10.3334/ORNLDAAC/1332](https://doi.org/10.3334/ORNLDAAC/1332)) | manual download | `data/Circumpolar_Thermokarst_Landscapes/Circumpolar_Thermokarst_Landscapes.shp` | Fig 10 |
| EPA Level III Ecoregions of Alaska ([EPA](https://www.epa.gov/eco-research/ecoregion-download-files-state-region-10)) | manual download | `data/ak_eco_l3/ak_eco_l3.shp` | Fig 9 |

## Reproduce

Run everything with `poetry run python <script>` from the repository root.

**1. Features** (skip unless rebuilding; outputs are frozen)

| Step | Script | Output |
| --- | --- | --- |
| 1a | `data/fetch_alfresco.py` | `data/alfresco/*.tif` |
| 1b **GEE** | `data/build_daymet_rasters.py` | Daymet raster |
| 1c **GEE** | `data/build_modis_fire_rasters.py` | MODIS fire raster |
| 1d **GEE** | `data/build_feature_table.py` (~12 h) | `data/features_dirty.csv` (tracked) |
| 1e **GEE** | `data/build_prediction_data.py` | `data/prediction_data.nc` |

To skip 1e, download `prediction_datacube.nc` from the data record and save it as `data/prediction_data.nc`. It has the same schema.

**2. Model, map, and diagnostics** (no GEE; needs `data/prediction_data.nc` and the Obu raster from step 4 on)

1. `data/clean_feature_table.py` → `data/features_clean.csv`
2. `models/train_xgboost.py` → `models/model.json`, `cv_config.json`, `selected_hparams.json`, `run_manifest.json`
3. `diagnostics/repeated_cv.py`, `diagnostics/extrapolation_range.py` → JSON inputs for Fig 5
4. `diagnostics/feature_provenance.py`, `diagnostics/block_cv.py` (printed checks)
5. `models/shap_values.py`, `models/shap_groups.py`, `models/shap_mechanism_cache.py`
6. `models/predict.py` → `data/susceptibility.nc`
7. `diagnostics/aoa_calibration.py` → `models/aoa_threshold.json`
8. `models/aoa.py` → `data/aoa.nc`
9. `models/shap_dominance_cache.py`
10. `data/export_zenodo.py` → `data/zenodo/`

**3. Figures**

Run `output/fig05_cache_build.py`, then each `output/fig*.py` and `output/render_family_dendrogram.py`. Figure 1a is drawn in `output/process-schematic-revised.pptx` and exported by hand to `output/fig01_schematic_panel.pdf`.

## Licenses

- Code: GPL-3.0-or-later (`LICENSE`), including the trained model in `models/model.json`.
- Data record: CC-BY-4.0.
- Alaska Permafrost Thaw Database: CC0.
- Source Sans 3 fonts: SIL Open Font License 1.1 (`output/fonts/LICENSE.md`).
- Field photos in `data/photos/`: courtesy of Irina Overeem and Kevin Rozmiarek.
