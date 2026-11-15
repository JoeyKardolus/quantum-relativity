"""Contact table: the cells' neighbour lists with face counts, laid out for the GPU.

Contract: `contact_table(owner, cell_count, slots)` returns (neighbour, faces, width):
two int32 numpy arrays of cell_count * width entries and the row width, which is
`slots`, or the next power of two above it that fits the most crowded row (a volume
that many joints fell into touches many joints). For each touching pair (low, high)
from contact_pairs.py, one slot of row `low` (entries low * width to low * width +
width - 1) holds `high` in `neighbour` and the shared face count in `faces`; a row's
pairs fill its first slots, unused slots hold -1 and 0. Each pair is stored once, in
its lower cell's row, so a box move finds a pair by scanning one row up to its first
empty slot and changes its count with one atomic add (box_moves.py).
"""
from __future__ import annotations

import numpy as np

from physics.potts.contact_pairs import contact_pairs


def contact_table(owner: np.ndarray, cell_count: int, slots: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(neighbour, faces, width): rows per cell, each touching pair once, in its lower cell's row."""
    low, high, faces = contact_pairs(owner)
    order = np.argsort(low, kind="stable")
    low, high, faces = low[order], high[order], faces[order]
    rank = np.arange(low.size) - np.searchsorted(low, low, side="left")
    width = int(slots)
    while low.size and int(rank.max()) >= width:
        width *= 2
    neighbour = np.full(cell_count * width, -1, dtype=np.int32)
    counts = np.zeros(cell_count * width, dtype=np.int32)
    place = low * width + rank
    neighbour[place] = high
    counts[place] = faces
    return neighbour, counts, width
