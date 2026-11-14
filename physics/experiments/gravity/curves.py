"""Per-step curves read from lump records.

Contract: `lump_curves(records, ruler, unit)` returns, one value per step:
  "centre"      lump-centre distance in far-background mean lengths, divided
                by its step-0 value (1 = no approach; a whole-space shrink or
                growth leaves it at 1);
  "joint"       the same for the two starting joints (the control has no
                lumps, so this is its curve);
  "raw"         lump-centre distance in starting mean lengths `unit`;
  "background"  far-background mean connection length / its step-0 value;
  "fixed"       lump-centre separation on the fixed ruler (undisturbed
                network, mean lengths), which moves only when a centre moves
                to other joints; a centre that is a merged joint stands at the
                mean ruler value of the original joints in it.
"""
from __future__ import annotations

import numpy as np


def lump_curves(records: list[dict], ruler: np.ndarray, unit: float) -> dict:
    """Background-relative, raw, background-size and fixed-ruler curves per step."""
    background = np.array([record["background_length"] for record in records])
    return {"centre": _relative(records, "centre_distance", background),
            "joint": _relative(records, "joint_distance", background),
            "raw": np.array([record["centre_distance"] for record in records]) / unit,
            "background": background / background[0],
            "fixed": np.array([ruler[0, record["origin"][1]].mean() + ruler[1, record["origin"][0]].mean()
                               for record in records]) / 2}


def _relative(records: list[dict], key: str, background: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.array([record[key] for record in records]) / background
        return values / values[0]
