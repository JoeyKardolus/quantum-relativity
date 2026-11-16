"""Kuhn corner (Warp func): which corner of its cell each corner of a Kuhn tetrahedron is.

Contract: `kuhn_corner(tet, slot)` returns the corner bits (dx, dy, dz), each 0 or 1, of
corner `slot` (0..3) of tetrahedron `tet` (0..5) of a lattice cell (kuhn_tet.py): tet t
runs from 000 along the axes in the order of permutation t ((x y z), (x z y), (y x z),
(y z x), (z x y), (z y x)) to 111, its slots ordered so an unmoved cell's tet has
positive volume (slots 1 and 2 swapped for the odd permutations 1, 2 and 5).
"""
from __future__ import annotations

import warp as wp


@wp.func
def _first_axis(tet: int) -> int:
    return tet // 2


@wp.func
def _second_axis(tet: int) -> int:
    first = tet // 2
    if first == 1:
        if tet % 2 == 0:
            return 0
        return 2
    return (first + 1 + tet % 2) % 3


@wp.func
def kuhn_corner(tet: int, slot: int) -> wp.vec3i:
    """The cell-corner bits of corner `slot` of Kuhn tetrahedron `tet`, positively ordered."""
    odd = tet == 1 or tet == 2 or tet == 5
    walked = slot
    if odd and slot == 1:
        walked = 2
    elif odd and slot == 2:
        walked = 1
    bits = wp.vec3i(0, 0, 0)
    if walked == 3:
        return wp.vec3i(1, 1, 1)
    if walked >= 1:
        bits[_first_axis(tet)] = 1
    if walked >= 2:
        bits[_second_axis(tet)] = 1
    return bits
