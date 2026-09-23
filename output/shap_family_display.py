"""Display names, member lines, and colors for the SHAP emergent families."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from cmcrameri import cm
from matplotlib.colors import to_hex

import figstyle

FAMILIES_JSON = Path(__file__).resolve().parent / "shap_families.json"

NAMES = {
    "Alpine relief": "Relief and snowpack",
    "Annual / dry-season temperature": "Mean annual temperature",
    "Land Cover (18 classes)": "Land cover",
    "Thermal continentality": "Continentality",
    "Isothermality / precip seasonality": "Isothermality",
    "Precipitation amount": "Annual precipitation",
    "Trend in precipitation": "Trend in precipitation",
    "Summer warmth": "Summer temperature",
    "Mean curvature (500 m)": "Mean curvature (500 m)",
    "Upstream Area": "Upstream area",
    "Sand fraction": "Sand fraction",
    "Mean curvature (2 km)": "Mean curvature (2 km)",
    "Soil organic / fertility": "Soil carbon and nitrogen",
    "Clay fraction": "Clay fraction",
    "Bulk density": "Soil bulk density",
    "Vegetation Mode (7 classes)": "Vegetation mode",
    "Flammability Index": "Flammability index",
    "Yedoma": "Yedoma",
    "Trend in temperature": "Trend in temperature",
    "Eastness": "Eastness",
    "Northness": "Northness",
    "Fire history": "Fire history",
}

MEMBERS = {
    "Alpine relief": ("elevation", "slope", "HAND", "mean annual SWE"),
    "Annual / dry-season temperature": ("MAAT", "mean temp. of driest quarter"),
    "Thermal continentality": ("mean diurnal range", "temp. seasonality",
                               "temp. annual range", "min temp. of coldest month",
                               "mean temp. of coldest quarter", "trend in SWE"),
    "Isothermality / precip seasonality": ("isothermality", "precip. seasonality"),
    "Precipitation amount": ("annual precip.", "precip. of wettest and driest month",
                             "precip. of wettest, driest, warmest, and coldest quarter"),
    "Summer warmth": ("max temp. of warmest month", "mean temp. of warmest quarter",
                      "mean temp. of wettest quarter"),
    "Sand fraction": ("sand 0–30 cm", "sand 30–200 cm"),
    "Soil organic / fertility": ("SOC 0–30 cm", "SOC 30–200 cm", "N 0–30 cm", "N 30–200 cm"),
    "Clay fraction": ("clay 0–30 cm", "clay 30–200 cm"),
    "Bulk density": ("bulk density 0–30 cm", "bulk density 30–200 cm"),
    "Fire history": ("time since last fire", "burn count"),
}

_S = [to_hex(cm.batlowS(i)) for i in range(8)]
COLORS = {
    "Annual / dry-season temperature": _S[4],
    "Summer warmth": "#E1A100",
    "Alpine relief": _S[3],
    "Precipitation amount": _S[0],
    "Mean curvature (500 m)": _S[5],
    "Thermal continentality": "#b07c8c",
    "Sand fraction": _S[2],
}


def color(label):
    return COLORS.get(label, figstyle.OTHER_GRAY)


def family_of_features(feature_names):
    families = json.loads(FAMILIES_JSON.read_text())["families"]
    lookup = {m: f["label"] for f in families for m in f["members"]}
    return np.array([lookup[f] for f in feature_names], dtype=object)


if __name__ == "__main__":
    print(figstyle.assert_cvd_safe([*COLORS.values(), figstyle.OTHER_GRAY]))
