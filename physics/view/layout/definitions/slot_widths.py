"""Slot widths: how wide every place of the pattern is along x, y and z.

Contract: `slot_widths(along, across, slot_axis, first=(0, 0, 0))` takes three arrays
over a box of lattice places with one more place all round it (so (nx + 2, ny + 2,
nz + 2); the grid wraps, so the place before 0 is the last one): per slot the length of
what is drawn in it along its stripe axis (`along`: the cell's length, or the summed
lengths of the cells a tear left there), across it (`across`) and that axis
(`slot_axis`). `first` is the lattice place of the box's first slot. It returns float64
(3, nx, ny, nz), widths[axis][i, j, k] for the box without the extra places.
A cell has a length and no shape, so the pattern gives the shape:
  - along an axis where the slot's index is odd, or for an energy volume along every
    axis, the slot is as wide as its cell is long (a face joint is its own length one
    way, an edge joint two ways, a corner joint all three)
  - along an axis where a joint only lies beside energy volumes, it is as wide as the
    mean of the two slots beside it, so a joint is as long as the volumes it joins
"""
from __future__ import annotations

import numpy as np


def slot_widths(along: np.ndarray, across: np.ndarray, slot_axis: np.ndarray, first: tuple = (0, 0, 0)) -> np.ndarray:
    """Every slot's width along each axis."""
    shape = along.shape
    odd = [((np.arange(shape[axis]) + first[axis] - 1) % 2 == 1).reshape([-1 if other == axis else 1 for other in range(3)])
           for axis in range(3)]
    widths = np.empty((3, shape[0] - 2, shape[1] - 2, shape[2] - 2))
    for axis in range(3):
        wide = np.where(slot_axis == axis, along, across).astype(np.float64)
        for other in range(3):
            if other != axis:
                beside = 0.5 * (np.roll(wide, 1, axis=other) + np.roll(wide, -1, axis=other))
                wide = np.where(odd[other] & ~odd[axis], beside, wide)
        widths[axis] = wide[1:-1, 1:-1, 1:-1]
    return widths
