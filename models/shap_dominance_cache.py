"""Per-cell dominant SHAP family for Figure 9.

For every in-AOA cell: TreeSHAP over the all-data model, sum signed SHAP within each
emergent family, and take the family with the largest |net contribution|.

    poetry run python models/shap_dominance_cache.py

Writes output/shap_dominance_cache.npz (multi-minute). SHAP_DOM_SMOKE=1 subsamples
in-AOA cells and writes to output/_smoke/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT  # noqa: E402

sys.path.insert(0, str(OUTPUT))
import shap_family_display as fd  # noqa: E402

MODEL_PATH = MODELS / "model.json"
PRED_NC = DATA / "prediction_data.nc"
AOA_NC = DATA / "aoa.nc"

SMOKE = bool(os.environ.get("SHAP_DOM_SMOKE"))
SMOKE_N = 60_000
CHUNK = 200_000
GATE_FRACTION = 0.60


def load_feature_stack():
    """Return (feature_array (n_cells, 70), (ny, nx), feature_names) in MODEL order."""
    model_feature_names = json.loads(MODEL_PATH.read_text())["learner"]["feature_names"]

    ds = xr.open_dataset(PRED_NC)
    stack = ds["feature_stack"].values
    ds_names = ds["feature"].values.tolist()
    default_value = ds.attrs.get("default_value", -9999)
    ny, nx, nf = stack.shape

    if ds_names != model_feature_names:
        order = [ds_names.index(n) for n in model_feature_names]
        stack = stack[:, :, order]

    arr = stack.reshape(ny * nx, nf).astype(np.float32)
    arr[arr == default_value] = np.nan
    return arr, (ny, nx), model_feature_names


def load_in_aoa(shape):
    ny, nx = shape
    aoa = xr.open_dataset(AOA_NC)["inside_aoa"].values
    if aoa.shape != (ny, nx):
        raise ValueError(f"aoa.nc grid {aoa.shape} != datacube grid {(ny, nx)}")
    return (aoa == 1).reshape(ny * nx)


def dominant_family(X, feature_names, booster, family_col_idx, n_families):
    n = X.shape[0]
    codes = np.empty(n, dtype=np.int8)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        dm = xgb.DMatrix(X[start:stop], feature_names=feature_names)
        contribs = booster.predict(dm, pred_contribs=True)[:, :-1]
        net = np.zeros((stop - start, n_families), dtype=np.float64)
        np.add.at(net.T, family_col_idx, contribs.T)
        codes[start:stop] = np.argmax(np.abs(net), axis=1)
        print(f"    SHAP {stop:>9,}/{n:,} cells", flush=True)
    return codes


def main():
    print(f"{'[SMOKE] ' if SMOKE else ''}Figure 9 dominant-family cache")
    arr, (ny, nx), feature_names = load_feature_stack()
    in_aoa = load_in_aoa((ny, nx))
    idx_in_aoa = np.nonzero(in_aoa)[0]
    print(f"grid {ny}x{nx} = {ny*nx:,} cells | in-AOA {idx_in_aoa.size:,}")

    if SMOKE:
        rng = np.random.default_rng(42)
        sel = rng.choice(idx_in_aoa.size, size=min(SMOKE_N, idx_in_aoa.size),
                         replace=False)
        idx_in_aoa = np.sort(idx_in_aoa[sel])
        print(f"[SMOKE] subsampled to {idx_in_aoa.size:,} in-AOA cells")

    X = arr[idx_in_aoa]

    families = list(fd.NAMES)
    family_of_feature = fd.family_of_features(feature_names)
    fam_to_col = {f: i for i, f in enumerate(families)}
    family_col_idx = np.array([fam_to_col[f] for f in family_of_feature], dtype=np.intp)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    booster = model.get_booster()

    print(f"computing per-cell SHAP over {len(feature_names)} features, "
          f"{len(families)} families, chunks of {CHUNK:,} ...")
    codes = dominant_family(X, feature_names, booster, family_col_idx, len(families))

    fam_raster = np.full(ny * nx, -1, dtype=np.int8)
    fam_raster[idx_in_aoa] = codes
    fam_raster = fam_raster.reshape(ny, nx)

    counts = np.bincount(codes, minlength=len(families))
    fracs = counts / counts.sum()
    print("\n% of scored in-AOA cells each family dominates:")
    for i in np.argsort(fracs)[::-1]:
        print(f"  {fracs[i]*100:6.2f}%  {counts[i]:>9,}  {fd.NAMES[families[i]]}")

    top_frac = float(fracs.max())
    top_fam = families[int(np.argmax(fracs))]
    gate_ok = top_frac <= GATE_FRACTION
    print(f"\nVALIDATION GATE: top family '{fd.NAMES[top_fam]}' = {top_frac*100:.1f}% "
          f"(threshold {GATE_FRACTION*100:.0f}%) -> {'PASS' if gate_ok else 'FAIL'}")

    out_dir = OUTPUT / "_smoke" if SMOKE else OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "shap_dominance_cache.npz"
    np.savez(
        out,
        dominant_family=fam_raster,
        families=np.array(families, dtype=object),
        area_counts=counts.astype(np.int64),
        area_fraction=fracs.astype(np.float64),
        n_in_aoa=np.int64(idx_in_aoa.size),
        gate_top_family=np.array(top_fam, dtype=object),
        gate_top_fraction=np.float64(top_frac),
        gate_ok=np.array(gate_ok),
        smoke=np.array(SMOKE),
    )
    print(f"\nwrote {out}  [raster {fam_raster.shape}, {idx_in_aoa.size:,} cells scored]")


if __name__ == "__main__":
    main()
