from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
OUTPUT = ROOT / 'output'

# --------------------------------------------------------------------------
# Google Earth Engine identity (single source of truth for the GEE scripts)
# --------------------------------------------------------------------------
# EE_PROJECT is the *compute* project passed to ee.Initialize(project=...).
#
# There is intentionally NO ASSET_ROOT / custom-asset path any more. Access to
# the original `ee-abrupt-thaw` project was lost (2026-07-10) and all 13 old
# custom assets were confirmed unreadable, so the feature side was rebuilt with
# ZERO custom uploaded assets (TASKS T0): the GEE track computes inline from
# public catalog datasets (data/gee_features.py) and the LOCAL track samples
# downloaded rasters in Python (data/local_rasters.py). This ends the
# project-scoped-asset fragility for good — do not reintroduce an ASSET_ROOT.
EE_PROJECT = 'abrupt-thaw-indicators'