"""Place cell: the id of the energy volume at a place of the start lattice.

Contract: `place_cell(side, place)` takes the lattice's cells along an edge and a place
(i, j, k) and returns the cell id the engine's start gave it, `(i * side + j) * side + k`
(physics.engine start_grid). A place is an energy volume only where all three indices
are even. Raises ValueError for an odd index or one outside 0 .. side - 1. The place is
a name for a start cell, nothing more: after the start a cell has only relations.
"""
from __future__ import annotations


def place_cell(side: int, place: tuple) -> int:
    """The id of the energy volume at lattice place (i, j, k)."""
    if len(place) != 3 or any(index % 2 or not 0 <= index < side for index in place):
        raise ValueError(f"{place} is not an energy volume of a {side}-place lattice")
    return (int(place[0]) * side + int(place[1])) * side + int(place[2])
