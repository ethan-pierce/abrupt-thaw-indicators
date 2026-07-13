"""Fetch SNAP AR5/CMIP5 projected-climate decadal summaries for the three
`Projected ... change` features, replacing the lost custom GEE assets
(`summer-temperature-trend`, `winter-temperature-trend`,
`annual-precipitation-trend`) with a first-party, account-independent source.

Source
------
UAF SNAP, downscaled CMIP5/AR5 projections for Alaska at 771 m, EPSG:3338:
  https://data.snap.uaf.edu/data/Base/AK_771m/projected/AR5_CMIP5_models/
The `derived/*_decadal_summaries_*.zip` archives contain per-decade GeoTIFFs of
seasonal (JJA/DJF/MAM/SON) and annual means. GeoTIFF, mm (precip) / degrees C
(temperature). Downscaling baseline: PRISM 1971-2000.

Reconstruction decisions (2026-07-13; original build parameters were lost with
the `ee-abrupt-thaw` project — see [[ee-project-access-lost]] / TASKS T0, T29):
  * Model    : 5modelAvg (SNAP 5-model average)
  * Scenario : RCP 8.5
  * Product  : derived decadal summaries (not the 4.8 GB monthly series)
  * "Change" : end-century minus early-century, 2090_2099 - 2010_2019
  * Seasons  : summer = JJA, winter = DJF (SNAP's own precomputed aggregates)
These are DOCUMENTED reconstructions, not a byte-match to the lost assets; that
is acceptable for the v2.0.0 rebuild. Note the original asset label said
"CRU TS3.1" (the older AR4/CMIP3 2 km product); this AR5/CMIP5 771 m set is a
deliberate upgrade.

Feature definitions (computed downstream, e.g. in build_feature_table.py):
  Projected summer temperature change (C) = JJA(2090_2099) - JJA(2010_2019)
  Projected winter temperature change (C) = DJF(2090_2099) - DJF(2010_2019)
  Projected precipitation change    (mm)  = annual(2090_2099) - annual(2010_2019)

This script extracts ONLY the 6 needed decadal GeoTIFFs (~35 MB) out of the two
~650 MB archives, via HTTP range requests against the zip central directory, so
no full archive download is required. Output: settings.DATA/snap/*.tif
(git-ignored; regenerate by re-running). Idempotent: existing files are skipped.
"""

import io
import zipfile
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA

OUT_DIR = DATA / 'snap'

_BASE = ('https://data.snap.uaf.edu/data/Base/AK_771m/projected/'
         'AR5_CMIP5_models')
_TEMP_ZIP = (f'{_BASE}/Projected_Monthly_and_Derived_Temperature_Products_'
             f'771m_CMIP5_AR5/derived/'
             f'tas_decadal_summaries_AK_771m_5modelAvg_rcp85.zip')
_PRECIP_ZIP = (f'{_BASE}/Projected_Monthly_and_Derived_Precipitation_Products_'
               f'771m_CMIP5_AR5/derived/'
               f'pr_decadal_summaries_AK_771m_5modelAvg_rcp85.zip')

# member-in-zip -> output filename (output name == basename)
MEMBERS = {
    _TEMP_ZIP: [
        'decadal_mean/tas_decadal_mean_JJA_mean_c_5modelAvg_rcp85_2010_2019.tif',
        'decadal_mean/tas_decadal_mean_JJA_mean_c_5modelAvg_rcp85_2090_2099.tif',
        'decadal_mean/tas_decadal_mean_DJF_mean_c_5modelAvg_rcp85_2010_2019.tif',
        'decadal_mean/tas_decadal_mean_DJF_mean_c_5modelAvg_rcp85_2090_2099.tif',
    ],
    _PRECIP_ZIP: [
        'decadal_mean/pr_decadal_mean_annual_total_mm_5modelAvg_rcp85_2010_2019.tif',
        'decadal_mean/pr_decadal_mean_annual_total_mm_5modelAvg_rcp85_2090_2099.tif',
    ],
}


class _HTTPRangeReader(io.RawIOBase):
    """Seekable file-like object backed by HTTP Range GETs, so zipfile can read
    a remote zip's central directory and extract single members without
    downloading the whole archive."""

    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers['Content-Length'])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            end = self.size - 1
        else:
            end = min(self.pos + n - 1, self.size - 1)
        if self.pos > end:
            return b''
        req = urllib.request.Request(
            self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        self.pos += len(data)
        return data


def fetch(out_dir: Path = OUT_DIR) -> list[Path]:
    """Extract the 6 SNAP decadal GeoTIFFs into ``out_dir``. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for url, members in MEMBERS.items():
        zf = None
        for member in members:
            out = out_dir / Path(member).name
            if out.exists():
                print(f'skip (exists) {out.name}')
                written.append(out)
                continue
            if zf is None:  # open lazily: only touch the network if needed
                zf = zipfile.ZipFile(_HTTPRangeReader(url))
            with zf.open(member) as src:
                data = src.read()
            out.write_bytes(data)
            print(f'wrote {out.name} ({len(data) / 1e6:.1f} MB)')
            written.append(out)
    return written


if __name__ == '__main__':
    paths = fetch()
    print(f'\n{len(paths)} SNAP decadal GeoTIFFs in {OUT_DIR}')
