"""Slot widths: how wide every place of the pattern is along x, y and z.

Contract: `slot_widths(side, along, across, slot_axis)` takes, per slot (side ** 3),
the length of what is drawn in it along its stripe axis (`along`: the cell's length, or
the summed lengths of the cells a tear left there), across it (`across`) and that axis
(`slot_axis`). It returns float64 (3, side, side, side), widths[axis][i, j, k].
A cell has a length and no shape, so the pattern gives the shape:
  - along an axis where the slot's index is odd, or for an energy volume along every
    axis, the slot is as wide as its cell is long (a face joint is its own length one
    way, an edge joint two ways, a corner joint all three)
  - along an axis where a joint only lies beside energy volumes, it is as wide as the
    mean of the two slots beside it, so a joint is as long as the volumes it joins
"""
from __future__ import annotations

import numpy as np


def slot_widths(side: int, along: np.ndarray, across: np.ndarray, slot_axis: np.ndarray) -> np.ndarray:
    """Every slot's width along each axis."""
    odd = np.indices((side, side, side)) % 2 == 1
    widths = np.zeros((3, side, side, side))
    for axis in range(3):
        wide = np.where(slot_axis == axis, along, across).reshape(side, side, side).astype(np.float64)
        for other in range(3):
            if other != axis:
                beside = 0.5 * (np.roll(wide, 1, axis=other) + np.roll(wide, -1, axis=other))
                wide = np.where(odd[other] & ~odd[axis], beside, wide)
        widths[axis] = wide
    return widths
