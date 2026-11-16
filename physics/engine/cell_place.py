"""Cell place (Warp func): a cell's three lattice indices from its id, in the extended lattice.

Contract: `cell_place(cell, extent)` returns the vec3i (a, b, c) of cell id
(a * extent[1] + b) * extent[2] + c, extent the batch's extended cells per axis
(state.extent): the forced cell lattice (state.py) holds cells[axis] real cells along
each axis, indices 1 .. cells[axis], inside a ring of empty cells, indices 0 and
cells[axis] + 1, that covers the boxes round it out to the universe's walls. A universe
whose lattice has fewer cells than the batch's storage leaves the indices past its
ring unused (cell_kind.py: they are ring too, of no volume). `cell_id` is the inverse.
"""
from __future__ import annotations

import warp as wp


@wp.func
def cell_place(cell: int, extent: wp.vec3i) -> wp.vec3i:
    """The lattice indices (a, b, c) of an extended-lattice cell id."""
    return wp.vec3i(cell // (extent[1] * extent[2]), (cell // extent[2]) % extent[1], cell % extent[2])


@wp.func
def cell_id(place: wp.vec3i, extent: wp.vec3i) -> int:
    """The extended-lattice cell id of lattice indices (a, b, c)."""
    return (place[0] * extent[1] + place[1]) * extent[2] + place[2]
