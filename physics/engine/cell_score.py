"""Cell score (Warp func): how far inside a moving lattice cell a point lies.

Contract: `cell_score(corners, home, cell, extent, point)` returns, for the
extended-lattice cell id `cell` of universe `home` and a point in that universe's box
coordinates, the largest over the cell's six Kuhn tetrahedra (kuhn_tet.py) of the
smallest barycentric coordinate of the point in that tet: >= 0 when the point lies in
the cell, negative outside, the more negative the farther. A tet squashed flat scores
-1e30 (it holds nothing). box_targets.py reads it to give each Planck box the cell
whose moving region covers the box's centre.
"""
from __future__ import annotations

import warp as wp

from physics.engine.cell_place import cell_place
from physics.engine.kuhn_tet import kuhn_tet


@wp.func
def cell_score(corners: wp.array3d(dtype=wp.vec3), home: int, cell: int, extent: wp.vec3i,
               point: wp.vec3) -> float:
    """The point's depth inside the cell: max over its tets of the smallest barycentric coordinate."""
    place = cell_place(cell, extent)
    best = float(-1.0e30)
    for tet in range(6):
        start, one, two, end = kuhn_tet(corners, home, place, tet, extent)
        frame = wp.mat33(one[0] - start[0], two[0] - start[0], end[0] - start[0],
                         one[1] - start[1], two[1] - start[1], end[1] - start[1],
                         one[2] - start[2], two[2] - start[2], end[2] - start[2])
        if wp.determinant(frame) > 1.0e-9:
            weights = wp.inverse(frame) * (point - start)
            smallest = wp.min(wp.min(weights[0], weights[1]), wp.min(weights[2], 1.0 - weights[0] - weights[1]
                                                                     - weights[2]))
            best = wp.max(best, smallest)
    return best
