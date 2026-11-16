"""Lattice kinds: every cell's kind in the forced cell lattice, on the host (numpy mirror of cell_kind.py).

Contract: `lattice_kinds(cells_per_side)` returns an int8 array of shape (m, m, m), m =
cells_per_side + 2, indexed by the extended lattice indices (a, b, c) (cell_place.py):
1 energy volume (all three indices even, in 2 .. cells_per_side - 1), -1 joint (every
other real cell), 0 the ring of empty space round it. `.ravel()` gives it per cell id.
Raises ValueError for fewer than 3 cells a side.
"""
from __future__ import annotations

import numpy as np


def lattice_kinds(cells_per_side: int) -> np.ndarray:
    """Per extended-lattice cell: 1 energy volume, -1 joint, 0 ring."""
    if cells_per_side < 3:
        raise ValueError(f"a cell lattice needs at least 3 cells a side, got {cells_per_side}")
    index = np.arange(cells_per_side + 2)
    real = (index >= 1) & (index <= cells_per_side)
    volume = (index % 2 == 0) & (index >= 2) & (index <= cells_per_side - 1)
    inside = real[:, None, None] & real[None, :, None] & real[None, None, :]
    volumes = volume[:, None, None] & volume[None, :, None] & volume[None, None, :]
    return np.where(volumes, 1, np.where(inside, -1, 0)).astype(np.int8)
