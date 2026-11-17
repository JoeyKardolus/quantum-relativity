"""Lattice step (Warp func): the lattice place one step from another in one of the 26 directions, round the edge.

Contract: `lattice_step(cell, side, code)` returns the id of the lattice place reached
from place `cell` (id (a * side + b) * side + c) by the step dx = code // 9 - 1,
dy = (code // 3) % 3 - 1, dz = code % 3 - 1, each index taken round `side` (the lattice
wraps). Code 13 is the place itself. Only pristine cells' relations are read this way
(state.py).
"""
from __future__ import annotations

import warp as wp


@wp.func
def lattice_step(cell: int, side: int, code: int) -> int:
    """The lattice place in direction `code` from `cell`."""
    first = (cell // (side * side) + code // 9 - 1 + side) % side
    second = ((cell // side) % side + (code // 3) % 3 - 1 + side) % side
    third = (cell % side + code % 3 - 1 + side) % side
    return (first * side + second) * side + third
