"""Link table: every cell each cell touches, both ways round, laid out for the GPU.

Contract: `link_table(owner, cell_count, slots, pairs=None)` returns (links, width): an int32 numpy
array of cell_count * width entries and the row width, `slots` or the next power of two
above it that fits the most crowded row. Row c (entries c * width to c * width + width
- 1) lists, in its first slots, every cell sharing a box face with c (contact_pairs.py, or `pairs`, its result for
this owner);
unused slots hold -1. Unlike the contact table each pair is in both rows, so a box move
can walk all of a cell's links (box_moves.py: links bend, never stretch).
"""
from __future__ import annotations

import numpy as np

from physics.potts.contact_pairs import contact_pairs


def link_table(owner: np.ndarray, cell_count: int, slots: int, pairs: tuple | None = None) -> tuple[np.ndarray, int]:
    """(links, width): per cell, every cell it touches."""
    low, high, _ = contact_pairs(owner) if pairs is None else pairs
    rows = np.concatenate([low, high])
    partners = np.concatenate([high, low])
    order = np.argsort(rows, kind="stable")
    rows, partners = rows[order], partners[order]
    rank = np.arange(rows.size) - np.searchsorted(rows, rows, side="left")
    width = int(slots)
    while rows.size and int(rank.max()) >= width:
        width *= 2
    links = np.full(cell_count * width, -1, dtype=np.int32)
    links[rows * width + rank] = partners
    return links, width
