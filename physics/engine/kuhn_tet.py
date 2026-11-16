"""Kuhn tet (Warp func): one of the six tetrahedra a lattice cell is cut into, its corners read from the moving lattice.

Contract: `kuhn_tet(corners, home, place, tet, cells_per_side)` returns the four corner
positions (vec3, in box units of universe `home`'s own coordinates) of tetrahedron
`tet` (0..5) of the extended-lattice cell at `place` (cell_place.py), in the slot order
of kuhn_corner.py. A cell's eight corners are lattice vertices corners[home * (m + 1) +
a + dx, b + dy, c + dz] (m = cells_per_side + 2, dx, dy, dz in 0..1). Every cell is cut
the same way (Kuhn's triangulation), so two neighbouring cells cut their shared face
along the same diagonal and the cells tile space with no gap and no overlap however the
corners move, as long as no tet turns inside out (move_corners.py keeps them right
side out). An unmoved cell's tets have positive volume.
"""
from __future__ import annotations

import warp as wp

from physics.engine.kuhn_corner import kuhn_corner


@wp.func
def _vertex(corners: wp.array3d(dtype=wp.vec3), home: int, place: wp.vec3i, bits: wp.vec3i,
            cells_per_side: int) -> wp.vec3:
    rows = cells_per_side + 3
    return corners[home * rows + place[0] + bits[0], place[1] + bits[1], place[2] + bits[2]]


@wp.func
def kuhn_tet(corners: wp.array3d(dtype=wp.vec3), home: int, place: wp.vec3i, tet: int, cells_per_side: int):
    """The four corners of Kuhn tetrahedron `tet` of the cell at `place`, positively ordered."""
    start = _vertex(corners, home, place, kuhn_corner(tet, 0), cells_per_side)
    one = _vertex(corners, home, place, kuhn_corner(tet, 1), cells_per_side)
    two = _vertex(corners, home, place, kuhn_corner(tet, 2), cells_per_side)
    end = _vertex(corners, home, place, kuhn_corner(tet, 3), cells_per_side)
    return start, one, two, end
