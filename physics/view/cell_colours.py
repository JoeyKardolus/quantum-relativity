"""Cell colours: red energy volumes and blue joints, brighter the more packed, on one fixed scale.

Contract: `cell_colours(kind, packing)` returns uint8 (n, 3) RGB: an energy volume (kind
1) in VOLUME_RED, a joint (kind -1) in JOINT_BLUE, times a brightness that is DIMMEST
at 2 ** -OCTAVES times the start's packing or less, 1 at 2 ** OCTAVES times or more,
and straight in the logarithm between (the start's mean packing, 1, is halfway). The
scale is fixed: never ranked or stretched to the frame, so a colour means the same
packing in every frame.
"""
from __future__ import annotations

import numpy as np

VOLUME_RED = (255, 38, 26)
JOINT_BLUE = (40, 96, 255)
DIMMEST = 0.3
OCTAVES = 2.0


def cell_colours(kind: np.ndarray, packing: np.ndarray) -> np.ndarray:
    """RGB of every cell from its kind and its packing."""
    octaves = np.log2(np.maximum(np.asarray(packing, dtype=np.float64), 1e-30))
    bright = DIMMEST + (1.0 - DIMMEST) * np.clip((octaves + OCTAVES) / (2.0 * OCTAVES), 0.0, 1.0)
    base = np.where((np.asarray(kind) == 1)[:, None], np.array(VOLUME_RED, float), np.array(JOINT_BLUE, float))
    return np.round(base * bright[:, None]).astype(np.uint8)
