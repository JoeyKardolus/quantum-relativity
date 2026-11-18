"""Cell pieces: every living cell as the block of the warped grid it is drawn in.

Contract: `cell_pieces(side, kind, length, packing, lines=None, line_axis=None)` lays
the grid out (`cell_slots`, `cell_layout`) and returns a `Pieces` record, one row per
living cell. `kind`, `length` and `packing` have one row per cell id; `lines` and
`line_axis` are what `torn_lines` returns. A cell alone in its slot is the whole slot;
cells sharing a slot (a tear) are stripes of it along the slot's axis, as long as their
lengths.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.lattice_place import lattice_place
from physics.view.layout.definitions.pieces import Pieces
from physics.view.layout.logic.cell_layout import cell_layout
from physics.view.layout.logic.cell_slots import cell_slots


def cell_pieces(side: int, kind: np.ndarray, length: np.ndarray, packing: np.ndarray,
                lines: dict | None = None, line_axis: dict | None = None) -> Pieces:
    """Every living cell's block in the warped grid."""
    slots = cell_slots(side, kind, length, lines, line_axis)
    grid = cell_layout(slots["widths"])
    slot, cell = slots["piece_slot"], slots["piece_cell"]
    place = lattice_place(slot, side)
    corners = np.empty((len(slot), 8, 3))
    for corner in range(8):
        corners[:, corner] = grid[place[:, 0] + (corner & 1), place[:, 1] + ((corner >> 1) & 1),
                                  place[:, 2] + ((corner >> 2) & 1)]
    striped = (slots["piece_low"] > 0.0) | (slots["piece_high"] < 1.0)
    if striped.any():
        corners[striped] = _stripes(corners[striped], slots["slot_axis"][slot[striped]],
                                    slots["piece_low"][striped], slots["piece_high"][striped])
    return Pieces(corners.astype(np.float32), cell, np.asarray(kind)[cell].astype(np.int8),
                  np.asarray(packing, dtype=np.float32)[cell], place, int(side),
                  grid.reshape(-1, 3).min(axis=0).astype(np.float32),
                  grid.reshape(-1, 3).max(axis=0).astype(np.float32))


def _stripes(corners: np.ndarray, axis: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """The part of each block between shares `low` and `high` along its axis."""
    cut = np.empty_like(corners)
    rows = np.arange(len(corners))
    for corner in range(8):
        bit = 1 << axis
        near, far = corners[rows, corner & ~bit], corners[rows, corner | bit]
        share = np.where(corner & bit, high, low)[:, None]
        cut[:, corner] = near + (far - near) * share
    return cut
