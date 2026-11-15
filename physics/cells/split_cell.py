"""Split cell: one tetrahedron cut into four by the 1-4 move (bible rule 8, volume split).

Contract: `split_cell(corners, tets, cell)` returns (corners, tets, children, shares):
a new corner at the tetrahedron's centroid (appended as the last corner) and four
tetrahedra, each the parent with one of its corners replaced by the new one, so all
four stay positively oriented and fill the parent exactly. The first child takes
the parent's row, the other three are appended as the last rows; children (4,) are
their rows. shares (4,) are the children's measured volumes over their sum (each
1/4 to rounding), by which the caller shares the parent's energy and quanta, so
every child keeps the parent's density. The parent's four faces each go to one
child (so its neighbours keep their joints); the six new inner triangles are the
tear, new joints. No other tetrahedron is touched and the total space is kept.
A tetrahedron cannot be cut into two tetrahedra without a new corner on one of its
edges, which every tetrahedron around that edge shares: four is the least a split
that touches no neighbour can make.
"""
from __future__ import annotations

import numpy as np

from physics.cells.cell_volumes import cell_volumes


def split_cell(corners: np.ndarray, tets: np.ndarray, cell: int):
    """Corners and tetrahedra after the 1-4 move on `cell`, the children's rows and volume shares."""
    ring = tets[cell]
    centre = len(corners)
    grown = np.vstack([corners, corners[ring].mean(axis=0)])
    four = np.tile(ring, (4, 1))
    four[np.arange(4), np.arange(4)] = centre
    volumes = cell_volumes(grown, four)
    cut = tets.copy()
    cut[cell] = four[0]
    cut = np.vstack([cut, four[1:]])
    children = np.array([cell, len(tets), len(tets) + 1, len(tets) + 2], dtype=np.int64)
    return grown, cut, children, volumes / volumes.sum()
