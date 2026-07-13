"""Fetch UAF SNAP ALFRESCO historical spatial outputs for the two ALFRESCO
features, replacing the lost custom GEE assets
(`ALFRESCO-historical-flammability`, `ALFRESCO-historical-vegetation-mode`)
with a first-party, account-independent source.

Source
------
UAF SNAP, IEM/ALFRESCO Gen_1a relative spatial outputs:
  https://data.snap.uaf.edu/data/IEM/Outputs/ALF/Gen_1a/alfresco_relative_spatial_outputs/
Both products are GeoTIFF, EPSG:3338, 1 km, nodata = -9999.

Feature -> file (both the HISTORICAL, observed-climate run, matching the
original "*-historical-*" asset names; verified by inspection 2026-07-13):

  Flammability Index  (continuous, ~0-0.02; sample bilinear/nearest)
    relative_flammability/AR5_CMIP5/
      alfresco_relative_flammability_cru_ts40_historical_1900_1999_iem.tif
    -> CRU TS4.0 observed climate, 1900-1999.

  Vegetation Mode  (CATEGORICAL veg-type codes 0-8; sample NEAREST, never mean)
    vegetation_type/alfresco_vegetation_mode_statistic.zip ::
      alfresco_relative_vegetation_change_1950-2008_historical.tif
    -> modal vegetation type over the 1950-2008 historical run. NB the member
       filename says "..._change" but the values are discrete class codes
       (-9999 nodata + 0..8) -- it is the mode statistic (per the zip name).
       Legend: see relative_flammability/.../*_code.rtf and the product
       metadata on the SNAP portal.

Reconstruction note: original derivation params were lost with the
`ee-abrupt-thaw` project (see [[ee-project-access-lost]] / TASKS T0, T29); these
historical products are the documented reconstruction (OK for the v2.0.0
rebuild). Output: settings.DATA/alfresco/*.tif (git-ignored; regenerate by
re-running). Idempotent: existing files are skipped.
"""

import io
import zipfile
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA

OUT_DIR = DATA / 'alfresco'

_ALF = ('https://data.snap.uaf.edu/data/IEM/Outputs/ALF/Gen_1a/'
        'alfresco_relative_spatial_outputs')

# direct GeoTIFF downloads: url -> output basename
_DIRECT = {
    f'{_ALF}/relative_flammability/AR5_CMIP5/'
    f'alfresco_relative_flammability_cru_ts40_historical_1900_1999_iem.tif':
        'alfresco_relative_flammability_cru_ts40_historical_1900_1999_iem.tif',
}

# members to extract from a remote zip: zip-url -> {member: output basename}
_FROM_ZIP = {
    f'{_ALF}/vegetation_type/alfresco_vegetation_mode_statistic.zip': {
        'alfresco_relative_vegetation_change_1950-2008_historical.tif':
            'alfresco_vegetation_mode_1950-2008_historical.tif',
    },
}


class _HTTPRangeReader(io.RawIOBase):
    """Seekable file-like object backed by HTTP Range GETs, so zipfile can
    extract a single member without downloading the whole archive."""

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
    """Download/extract the 2 ALFRESCO historical GeoTIFFs. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for url, name in _DIRECT.items():
        out = out_dir / name
        if out.exists():
            print(f'skip (exists) {out.name}')
        else:
            urllib.request.urlretrieve(url, out)
            print(f'downloaded {out.name} ({out.stat().st_size / 1e6:.1f} MB)')
        written.append(out)

    for url, members in _FROM_ZIP.items():
        zf = None
        for member, name in members.items():
            out = out_dir / name
            if out.exists():
                print(f'skip (exists) {out.name}')
                written.append(out)
                continue
            if zf is None:
                zf = zipfile.ZipFile(_HTTPRangeReader(url))
            match = [n for n in zf.namelist() if n.endswith(member)]
            if not match:
                raise FileNotFoundError(f'{member} not in {url}')
            with zf.open(match[0]) as src:
                out.write_bytes(src.read())
            print(f'extracted {out.name} ({out.stat().st_size / 1e6:.1f} MB)')
            written.append(out)

    return written


if __name__ == '__main__':
    paths = fetch()
    print(f'\n{len(paths)} ALFRESCO historical GeoTIFFs in {OUT_DIR}')
