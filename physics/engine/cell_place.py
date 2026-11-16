"""Cell place (Warp func): a cell's three lattice indices from its id, in the extended lattice.

Contract: `cell_place(cell, cells_per_side)` returns the vec3i (a, b, c) of cell id
(a * m + b) * m + c, m = cells_per_side + 2: the forced cell lattice (state.py) holds
cells_per_side real cells a side, indices 1 .. cells_per_side, inside a ring of empty
cells, indices 0 and cells_per_side + 1, that covers the boxes round it out to the
universe's walls. `cell_id` is the inverse.
"""
from __future__ import annotations

import warp as wp


@wp.func
def cell_place(cell: int, cells_per_side: int) -> wp.vec3i:
    """The lattice indices (a, b, c) of an extended-lattice cell id."""
    extent = cells_per_side + 2
    return wp.vec3i(cell // (extent * extent), (cell // extent) % extent, cell % extent)
