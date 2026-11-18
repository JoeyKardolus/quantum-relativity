"""Cell pieces: cells as the blocks of the drawn grid.

Contract: `cell_pieces(side, kind, length, packing, lines=None, line_axis=None,
layer=None, only=None)` lays cells out (`typical_lengths`, `cell_slots`, `cell_layout`)
and returns a `Pieces` record. `kind`, `length` and `packing` have one row per cell id;
`lines` and `line_axis` are what `torn_lines` returns. A cell alone in its slot is the
whole slot; cells sharing a slot (a tear) are stripes of it along the slot's axis, as
long as their lengths. Which cells:
  layer k     the living cells of lattice layer k along z, laid out flat: only that
              layer and the two beside it on either side are read, whatever the size
              of the lattice
  layer None  the living cells for which `only` (bool per cell id) is true, or all of
              them when `only` is None. Only the box of places round those cells, plus
              REACH, is laid out. When that box holds more than LAID_OUT_MOST places
              nothing is laid out and `left_out` counts the cells.
`low` and `high` are the outer frame of the whole grid in every case.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.kinds import NO_CELL
from physics.view.layout.definitions.lattice_place import lattice_place
from physics.view.layout.definitions.pieces import Pieces
from physics.view.layout.definitions.warp_marks import ACROSS, LAID_OUT_MOST, REACH
from physics.view.layout.logic.cell_layout import cell_layout
from physics.view.layout.logic.cell_slots import cell_slots
from physics.view.layout.logic.typical_lengths import typical_lengths


def cell_pieces(side: int, kind: np.ndarray, length: np.ndarray, packing: np.ndarray,
                lines: dict | None = None, line_axis: dict | None = None, layer: int | None = None,
                only: np.ndarray | None = None) -> Pieces:
    """The blocks of one layer, or of the wanted cells, in the drawn grid."""
    kind = np.asarray(kind)
    typical = typical_lengths(kind, length, side ** 3)
    frame = np.full(3, 0.25 * side * (typical[0] + typical[1]), dtype=np.float32)
    flat = layer is not None
    if flat:
        box, reach = ((0, side), (0, side), (layer, layer + 1)), (0, 0, len(ACROSS) // 2)
    else:
        box, reach = _box_of(side, kind, only, lines or {}), (REACH, REACH, REACH)
    if box is None or (not flat and np.prod([min(high + REACH, side) - max(low - REACH, 0) for low, high in box])
                       > LAID_OUT_MOST):
        alive = kind != NO_CELL
        missed = 0 if box is None else int(np.count_nonzero(alive if only is None else alive & np.asarray(only)))
        return Pieces(np.zeros((0, 8, 3), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8),
                      np.zeros(0, dtype=np.float32), np.zeros((0, 3), dtype=np.int64), int(side), -frame, frame,
                      False, missed)
    slots = cell_slots(side, kind, length, lines, line_axis, box, reach)
    first = slots["first"]
    trim = tuple((low - start, start + slots["widths"].shape[1 + axis] - high)
                 for axis, ((low, high), start) in enumerate(zip(box, first)))
    grid, limited = cell_layout(slots["widths"], typical, first, side, flat, trim)
    slot, cell = slots["piece_slot"], slots["piece_cell"]
    if only is not None and not flat:
        keep = np.asarray(only)[cell]
        slot, cell = slot[keep], cell[keep]
        slots = {**slots, "piece_low": slots["piece_low"][keep], "piece_high": slots["piece_high"][keep]}
    place = lattice_place(slot, side)
    at = place - np.array([low for low, _ in box])
    corners = np.empty((len(slot), 8, 3))
    for corner in range(8):
        corners[:, corner] = grid[at[:, 0] + (corner & 1), at[:, 1] + ((corner >> 1) & 1), at[:, 2] + ((corner >> 2) & 1)]
    striped = (slots["piece_low"] > 0.0) | (slots["piece_high"] < 1.0)
    if striped.any():
        axis = np.array([slots["slot_axis"][one] for one in slot[striped].tolist()], dtype=np.int64)
        corners[striped] = _stripes(corners[striped], axis, slots["piece_low"][striped], slots["piece_high"][striped])
    return Pieces(corners.astype(np.float32), cell, kind[cell].astype(np.int8),
                  np.asarray(packing)[cell].astype(np.float32), place, int(side), -frame, frame, limited, 0)


def _box_of(side: int, kind: np.ndarray, only: np.ndarray | None, lines: dict) -> tuple | None:
    """The box of lattice places round the wanted cells (the whole lattice when all are), None when none is."""
    if only is None:
        return ((0, side),) * 3
    only = np.asarray(only)
    slots = np.flatnonzero(only[:side ** 3] & (kind[:side ** 3] != NO_CELL))
    torn = [slot for slot, living in lines.items() if only[living].any()]
    if not len(slots) and not torn:
        return None
    place = lattice_place(np.concatenate([slots, np.array(torn, dtype=np.int64)]), side)
    return tuple((int(low), int(high) + 1) for low, high in zip(place.min(axis=0), place.max(axis=0)))


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
