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
# These are deliberately DECOUPLED so ASSET_ROOT can lag EE_PROJECT during a
# migration. Access to the original `ee-abrupt-thaw` project was lost
# (2026-07-10); compute has moved to the new project `abrupt-thaw-indicators`
# (2026-07-13). On 2026-07-13 all 13 old custom assets were confirmed
# UNREADABLE from the new project (the old project itself is inaccessible:
# `earthengine.assets.list` denied), so the shared-readable shortcut does not
# apply — every custom asset must be re-sourced and re-uploaded under the new
# ASSET_ROOT. That re-upload is still pending (blocked on T29 for the SWE /
# climate-trend / curvature upstreams), so no custom asset resolves yet.
EE_PROJECT = 'abrupt-thaw-indicators'
ASSET_ROOT = f'projects/{EE_PROJECT}/assets'