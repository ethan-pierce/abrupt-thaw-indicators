from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
OUTPUT = ROOT / 'output'

# --------------------------------------------------------------------------
# Google Earth Engine identity (single source of truth for the GEE scripts)
# --------------------------------------------------------------------------
# EE_PROJECT is the *compute* project passed to ee.Initialize(project=...).
# ASSET_ROOT is the path prefix for this project's *custom uploaded* assets
# (curvature, ALFRESCO, FIRMS, SWE/climate trends, NLCD, ...). Public catalog
# datasets (USGS/3DEP, WORLDCLIM, projects/soilgrids-isric/*) are account-
# independent and are NOT routed through ASSET_ROOT.
#
# These are deliberately DECOUPLED: access to the original `ee-abrupt-thaw`
# project was lost (2026-07-10), so compute must move to a new project. If the
# old custom assets turn out to be shared-readable, point EE_PROJECT at the new
# project but leave ASSET_ROOT on `ee-abrupt-thaw` to keep reading them; if not,
# re-upload the assets under the new project and set ASSET_ROOT to match.
#
# TODO(migration): set EE_PROJECT to the new project id once it exists, and set
# ASSET_ROOT to wherever the custom assets end up living.
EE_PROJECT = 'ee-abrupt-thaw'
ASSET_ROOT = f'projects/{EE_PROJECT}/assets'