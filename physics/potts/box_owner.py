"""Box owner (Warp function): the cell owning a Planck box, -1 for an empty box or one outside the grid.

Contract: `box_owner(owner, i, j, k, side)` returns owner[i, j, k] when 0 <= i, j, k <
side and -1 otherwise, so the grid's outside reads as empty space (the cube is open:
nothing lies beyond, nothing wraps).
"""
from __future__ import annotations

import warp as wp


@wp.func
def box_owner(owner: wp.array3d(dtype=wp.int32), i: int, j: int, k: int, side: int) -> int:
    """The owner of box (i, j, k), -1 outside the grid."""
    if i < 0 or j < 0 or k < 0 or i >= side or j >= side or k >= side:
        return -1
    return owner[i, j, k]
