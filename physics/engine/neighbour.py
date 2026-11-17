"""Neighbour (Warp func): one of a cell's neighbours, from its own list or from the lattice.

Contract: `neighbour(cell, index, side, adj_at, adj_pool)` returns the id of neighbour
`index` (0 .. adj_n[cell] - 1) of `cell`: entry adj_at[cell] + index of the pool when
the cell keeps its own list (a tear or a collapse changed its relations, state.py),
else the lattice place in the index'th of the 26 directions (lattice_step.py).
"""
from __future__ import annotations

import warp as wp

from physics.engine.lattice_step import lattice_step


@wp.func
def neighbour(cell: int, index: int, side: int, adj_at: wp.array(dtype=wp.int32),
              adj_pool: wp.array(dtype=wp.int32)) -> int:
    """The id of the cell's neighbour number `index`."""
    at = adj_at[cell]
    if at >= 0:
        return adj_pool[at + index]
    code = index
    if code >= 13:
        code = index + 1
    return lattice_step(cell, side, code)
