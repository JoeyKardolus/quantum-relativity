"""Pieces: the drawn blocks of one moment of the grid.

Contract: `Pieces` is a frozen record, one row per laid-out cell:
  corners   float32 (n, 8, 3) the block's corners; corner number c lies on the high side
            along x when c & 1, along y when c & 2, along z when c & 4
  cell      int64 (n,) the cell's id
  kind      int8 (n,) 1 energy volume, -1 joint (kinds.py)
  packing   float32 (n,) the number the opacity is read from (`cell_colours`); the
            engine source hands the cell's packing over the start's mean of its kind
  place     int64 (n, 3) the lattice place of the slot it is drawn in
  side      lattice places per axis
  low, high float32 (3,) the corners of the grid's outer frame
  limited   True when a cell's size is further from the typical than the drawing shows
  left_out  cells asked for but not laid out: they span more places than are laid out
            at once
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pieces:
    """Laid-out cells as the blocks they are drawn in."""

    corners: np.ndarray
    cell: np.ndarray
    kind: np.ndarray
    packing: np.ndarray
    place: np.ndarray
    side: int
    low: np.ndarray
    high: np.ndarray
    limited: bool = False
    left_out: int = 0
