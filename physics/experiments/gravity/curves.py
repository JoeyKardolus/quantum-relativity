"""Per-step curves read from lump records.

Contract: `lump_curves(records, ruler)` returns {"centre": size-relative
lump-centre distance, "joint": size-relative starting-joint distance (both 1 at
step 0, divided by the network's size so a global shrink is not an approach),
"fixed": lump-centre separation on the fixed ruler (undisturbed network, mean
lengths), which moves only when a centre moves to other joints; a centre that
is a merged joint stands at the mean ruler value of the original joints in it}.
"""
from __future__ import annotations

import numpy as np


def lump_curves(records: list[dict], ruler: np.ndarray) -> dict:
    """Relative centre distance, relative joint distance and fixed-ruler separation per step."""
    return {"centre": _relative(records, "centre_distance"),
            "joint": _relative(records, "joint_distance"),
            "fixed": np.array([ruler[0, record["origin"][1]].mean() + ruler[1, record["origin"][0]].mean()
                               for record in records]) / 2}


def _relative(records: list[dict], key: str) -> np.ndarray:
    values = np.array([record[key] / record["scale"] for record in records])
    return values / values[0]
