"""Cell pieces: every living cell as the block of the warped grid it is drawn in.

Contract: `cell_pieces(side, kind, length, packing, torn_from=None)` lays the grid out
(`cell_slots`, `cell_layout`) and returns a Pieces record, one row per living cell:
  corners   float32 (n, 8, 3) the block's corners; corner number c lies on the high side
            along x when c & 1, along y when c & 2, along z when c & 4
  cell      int64 (n,) the cell's id; kind int8 (n,) 1 energy volume, -1 joint
  packing   float32 (n,) the engine's packing of the cell (1: as packed as the start)
  place     int64 (n, 3) the lattice place of the slot it is drawn in
  side      lattice places per axis; low, high float32 (3,) the grid's bounding corners
A cell alone in its slot is the whole slot; cells sharing a slot (a tear) are stripes of
it along the slot's axis, as long as their lengths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.view.cell_layout import cell_layout
from physics.view.cell_slots import cell_slots


@dataclass(frozen=True)
class Pieces:
    """The drawn blocks of one moment of the grid."""

    corners: np.ndarray
    cell: np.ndarray
    kind: np.ndarray
    packing: np.ndarray
    place: np.ndarray
    side: int
    low: np.ndarray
    high: np.ndarray


def cell_pieces(side: int, kind: np.ndarray, length: np.ndarray, packing: np.ndarray,
                torn_from: dict | None = None) -> Pieces:
    """Every living cell's block in the warped grid."""
    slots = cell_slots(side, kind, length, torn_from)
    grid = cell_layout(slots["widths"])
    slot, cell = slots["piece_slot"], slots["piece_cell"]
    place = np.stack([slot // (side * side), (slot // side) % side, slot % side], axis=1)
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
