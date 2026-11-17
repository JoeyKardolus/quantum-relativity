"""Cell colours: red energy volumes and blue joints, more solid the more packed, on one fixed scale.

Contract: `cell_colours(kind, packing)` returns uint8 (n, 4) RGBA. The hue never
changes: an energy volume (kind 1) is VOLUME_RED, a joint (kind -1) JOINT_BLUE. The
opacity (alpha / 255) carries the packing: packing / SOLID, 1 at SOLID times the
start's mean packing or more. So it is a straight line on a log-log scale, falls to
fully transparent as the packing goes to nothing, and a lump turns solid as it gathers
energy. The scale is fixed: never ranked or stretched to the frame, so an opacity
means the same packing in every frame. This is the opacity of one layer of cells over
black (the slice): the start's mean packing is faint but visible (0.25), twice it 0.5,
three times 0.75, four times solid.

In the whole grid the eye looks through two faces of every cell on the way, so a face
is drawn at this opacity to the power WHOLE_POWER: the mean 0.005 (32 places of even
cells hide a quarter of what lies behind them), twice the mean 0.07, three times 0.33,
four times solid.
"""
from __future__ import annotations

import numpy as np

VOLUME_RED = (255, 38, 26)
JOINT_BLUE = (40, 96, 255)
SOLID = 4.0
WHOLE_POWER = 3.8


def cell_colours(kind: np.ndarray, packing: np.ndarray) -> np.ndarray:
    """RGBA of every cell: the hue from its kind, the opacity from its packing."""
    opacity = np.clip(np.asarray(packing, dtype=np.float64) / SOLID, 0.0, 1.0)
    colours = np.empty((len(opacity), 4), dtype=np.uint8)
    colours[:, :3] = np.where((np.asarray(kind) == 1)[:, None], VOLUME_RED, JOINT_BLUE)
    colours[:, 3] = np.round(255.0 * opacity)
    return colours
