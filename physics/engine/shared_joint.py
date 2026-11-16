"""Shared joint (Warp func): the one joint cell two face-neighbouring energy volumes share, read off the lattice.

Contract: `shared_joint(first, second, extent, cells)` takes two extended-lattice cell
ids (cell_place.py) and the universe's real cells per axis, and returns the id of the
cell halfway between them when both are energy volumes (cell_kind.py) two lattice
steps apart along one axis and level on the other two: the one joint they share
(red | blue | red, owner 2026-11-16). Returns -1 for any other pair. Neighbours are
fixed by index: no box is read.
"""
from __future__ import annotations

import warp as wp

from physics.engine.cell_kind import cell_kind
from physics.engine.cell_place import cell_id, cell_place


@wp.func
def shared_joint(first: int, second: int, extent: wp.vec3i, cells: wp.vec3i) -> int:
    """The joint between two face-neighbouring energy volumes, else -1."""
    if cell_kind(first, extent, cells) != 1 or cell_kind(second, extent, cells) != 1:
        return -1
    one = cell_place(first, extent)
    two = cell_place(second, extent)
    apart = two - one
    if wp.abs(apart[0]) + wp.abs(apart[1]) + wp.abs(apart[2]) != 2:
        return -1
    if wp.abs(apart[0]) == 1 or wp.abs(apart[1]) == 1 or wp.abs(apart[2]) == 1:
        return -1
    middle = wp.vec3i(one[0] + apart[0] // 2, one[1] + apart[1] // 2, one[2] + apart[2] // 2)
    return cell_id(middle, extent)
