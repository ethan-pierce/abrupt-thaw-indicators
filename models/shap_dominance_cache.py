"""Per-cell dominant-domain cache for Figure 10 (SHAP dominant-domain map, L6c/L7).

For every in-AOA grid cell we ask: *which thematic domain moves the model's
prediction most at this location?* Concretely, per cell we run TreeSHAP over the
all-data model, sum each domain's per-feature contributions to a domain net
contribution, and take argmax over the eight domains of |net contribution|
(unsigned — we want which KIND of driver dominates, not its direction; direction
is Fig 4's job). This is the descriptive raster Fig 10 paints.

Why this is a separate cache script (mirrors models/shap_mechanism_cache.py):
the compute is heavy (per-cell TreeSHAP over ~2.77M cells) and must run once when
the model / datacube are final; the figure script (output/fig10_shap_dominance.py)
is then pure plotting and re-runs fast off the cache.

Key differences from the Fig 7/8 caches:
  * ALL-DATA model (models/model.json), scored over the prediction datacube —
    NOT the OOF fold-refit machinery on training points. Coherence with Fig 7's
    OOF SHAP is acceptable per STRATEGY.md; the DOMAIN DEFINITIONS are identical
    (both import output/shap_domains.py).
  * Orientation is IRRELEVANT here: we take |domain net SHAP|, so no Abrupt-
    orientation sign flip is needed (unlike Fig 7/8). We drop the bias column.

Inputs (paths verified 2026-07-23):
  * data/prediction_data.nc  — feature_stack (y, x, 70), EPSG:4326 1 km grid,
    per-cell lon/lat coords, -9999 fill. 3.9 GB.
  * models/model.json        — all-data XGBoost; feature order = learner.feature_names.
  * data/aoa.nc              — inside_aoa (1 = reliable, 0 = extrapolating,
    NaN = off-domain); == 1 already encodes the Obu domain AND the DI<=threshold
    AOA cut (2,773,804 cells).

Run once when the model / features are final:
    poetry run python models/shap_dominance_cache.py
Writes output/shap_dominance_cache.npz. Multi-minute run (per-cell TreeSHAP).
SHAP_DOM_SMOKE=1 subsamples in-AOA cells for a fast wiring check + an unbiased
area-fraction estimate (writes to output/_smoke/), so the VALIDATION GATE can be
checked before paying for the full compute.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import DATA, MODELS, OUTPUT  # noqa: E402

sys.path.insert(0, str(OUTPUT))
import shap_domains as sd  # noqa: E402  (single source of truth for domains)

MODEL_PATH = MODELS / "model.json"
PRED_NC = DATA / "prediction_data.nc"
AOA_NC = DATA / "aoa.nc"

SMOKE = bool(os.environ.get("SHAP_DOM_SMOKE"))
SMOKE_N = 60_000          # in-AOA cells to sample for the wiring / gate check
CHUNK = 200_000           # cells per TreeSHAP call (bounds peak memory)
GATE_FRACTION = 0.60      # validation gate: flag if one domain dominates > this


def load_feature_stack():
    """Return (feature_array (n_cells, 70), (ny, nx), feature_names) in MODEL order."""
    import json
    model_feature_names = json.loads(MODEL_PATH.read_text())["learner"]["feature_names"]

    ds = xr.open_dataset(PRED_NC)
    stack = ds["feature_stack"].values                    # (y, x, feature)
    ds_names = ds["feature"].values.tolist()
    default_value = ds.attrs.get("default_value", -9999)
    ny, nx, nf = stack.shape

    if ds_names != model_feature_names:                   # reorder to the model's order
        order = [ds_names.index(n) for n in model_feature_names]
        stack = stack[:, :, order]

    arr = stack.reshape(ny * nx, nf).astype(np.float32)
    arr[arr == default_value] = np.nan                    # -9999 fill -> NaN (XGBoost missing)
    return arr, (ny, nx), model_feature_names


def load_in_aoa(shape):
    """Boolean (n_cells,) mask of in-AOA cells (inside_aoa == 1)."""
    ny, nx = shape
    aoa = xr.open_dataset(AOA_NC)["inside_aoa"].values
    if aoa.shape != (ny, nx):
        raise ValueError(f"aoa.nc grid {aoa.shape} != datacube grid {(ny, nx)}")
    return (aoa == 1).reshape(ny * nx)


def dominant_domain(X, feature_names, booster, domain_col_idx, n_domains):
    """Per-cell argmax over domains of |summed per-feature SHAP| (chunked)."""
    n = X.shape[0]
    codes = np.empty(n, dtype=np.int8)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        dm = xgb.DMatrix(X[start:stop], feature_names=feature_names)
        contribs = booster.predict(dm, pred_contribs=True)   # (chunk, 71): 70 feats + bias
        contribs = contribs[:, :-1]                          # drop bias column
        # sum signed SHAP within each domain -> (chunk, n_domains)
        net = np.zeros((stop - start, n_domains), dtype=np.float64)
        np.add.at(net.T, domain_col_idx, contribs.T)
        codes[start:stop] = np.argmax(np.abs(net), axis=1)
        print(f"    SHAP {stop:>9,}/{n:,} cells", flush=True)
    return codes


def main():
    print(f"{'[SMOKE] ' if SMOKE else ''}Figure 10 dominant-domain cache")
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

    # feature -> domain-column index, in DOMAIN_ORDER (stable across both figures)
    domains = sd.DOMAIN_ORDER
    dom_of_feature = sd.domain_of_features(feature_names)
    dom_to_col = {d: i for i, d in enumerate(domains)}
    domain_col_idx = np.array([dom_to_col[d] for d in dom_of_feature], dtype=np.intp)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    booster = model.get_booster()

    print(f"computing per-cell SHAP over {len(feature_names)} features, "
          f"{len(domains)} domains, chunks of {CHUNK:,} ...")
    codes = dominant_domain(X, feature_names, booster, domain_col_idx, len(domains))

    # dominant-domain raster (int8): domain code in-AOA, -1 elsewhere
    dom_raster = np.full(ny * nx, -1, dtype=np.int8)
    dom_raster[idx_in_aoa] = codes
    dom_raster = dom_raster.reshape(ny, nx)

    # area fractions (unbiased over in-AOA cells; on smoke, over the subsample)
    counts = np.bincount(codes, minlength=len(domains))
    fracs = counts / counts.sum()
    order = np.argsort(fracs)[::-1]
    print("\narea-fraction table (% of scored in-AOA cells each domain dominates):")
    for i in order:
        print(f"  {fracs[i]*100:6.2f}%  {counts[i]:>9,}  {domains[i]}")

    # ---- VALIDATION GATE ----------------------------------------------------
    top_frac = float(fracs.max())
    top_dom = domains[int(np.argmax(fracs))]
    gate_ok = top_frac <= GATE_FRACTION
    print(f"\nVALIDATION GATE: top domain '{top_dom}' = {top_frac*100:.1f}% of area "
          f"(threshold {GATE_FRACTION*100:.0f}%) -> {'PASS' if gate_ok else 'FAIL — FLAG FOR HUMAN'}")
    if not gate_ok:
        print("  >>> One domain dominates a majority of in-AOA area: the map will be "
              "near-monochrome and the proxy-rebuttal framing is weakened. Per "
              "fig10_spec.md, STOP and rethink the figure's framing with the human "
              "before finalizing.")

    out_dir = OUTPUT / "_smoke" if SMOKE else OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "shap_dominance_cache.npz"
    np.savez(
        out,
        dominant_domain=dom_raster,                       # (ny, nx) int8, -1 = not in-AOA
        domains=np.array(domains, dtype=object),          # code -> domain name (DOMAIN_ORDER)
        area_counts=counts.astype(np.int64),              # cells dominated, per domain code
        area_fraction=fracs.astype(np.float64),           # fraction of scored in-AOA cells
        n_in_aoa=np.int64(idx_in_aoa.size),
        gate_top_domain=np.array(top_dom, dtype=object),
        gate_top_fraction=np.float64(top_frac),
        gate_ok=np.array(gate_ok),
        smoke=np.array(SMOKE),
    )
    print(f"\nwrote {out}  [raster {dom_raster.shape}, {idx_in_aoa.size:,} cells scored]")


if __name__ == "__main__":
    main()
