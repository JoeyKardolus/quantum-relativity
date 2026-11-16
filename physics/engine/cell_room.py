"""Cell room (Warp func): the share of a cell's size its boxes do not yet fill.

Contract: `cell_room(size, held)` returns (size - held) / max(size, 1): positive when the
cell holds fewer boxes than its size, negative when more. exchange.py and mortar_shift.py
move boxes towards the cell with the most room, so where space and the cells' sizes
disagree every cell is stretched or pressed by the same share of its size (boxes follow
sizes; a thin joint never soaks up the room a volume gives off).
"""
from __future__ import annotations

import warp as wp


@wp.func
def cell_room(size: float, held: int) -> float:
    """(size - held) / max(size, 1)."""
    return (size - float(held)) / wp.max(size, 1.0)
