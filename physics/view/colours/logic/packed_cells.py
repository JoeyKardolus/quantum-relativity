"""Packed cells: the cells packed enough to show in the whole grid.

Contract: `packed_cells(kind, packing, marks)` returns bool (n,): true for an energy
volume or a joint whose fill, on the whole grid's scale (`opacity_law` to WHOLE_POWER),
is at least one step of 255 solid. Every other cell is clear there, so the whole grid
neither lays it out nor draws it. `marks` are `start_marks`.
"""
from __future__ import annotations

import numpy as np

from physics.view.colours.definitions.palette import SOLID, WHOLE_POWER
from physics.view.layout.definitions.kinds import ENERGY_VOLUME, JOINT


def packed_cells(kind: np.ndarray, packing: np.ndarray, marks: dict) -> np.ndarray:
    """Which cells the whole grid shows."""
    kind, packing = np.asarray(kind), np.asarray(packing)
    least = 1.0 + (SOLID - 1.0) * (1.0 / 255.0) ** (1.0 / WHOLE_POWER)
    packed = np.zeros(len(kind), dtype=bool)
    for one in (ENERGY_VOLUME, JOINT):
        packed |= (kind == one) & (packing >= least * marks[one])
    return packed
