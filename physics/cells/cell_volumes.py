"""Cell volumes: each tetrahedron's measured volume, from the corners.

Contract: `cell_volumes(corners, tets)` returns (n,) per tetrahedron (a, b, c, d) its
signed volume det(b - a, c - a, d - a) / 6: positive while it is the right way out,
0 flat, negative inverted. The tetrahedra share corners and faces, so their volumes
fill the mesh with no gaps and no overlaps and sum to the mesh's volume.
"""
from __future__ import annotations

import numpy as np


def cell_volumes(corners: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Per tetrahedron its signed volume."""
    first, second, third, fourth = (corners[tets[:, column]] for column in range(4))
    return np.einsum("ij,ij->i", second - first, np.cross(third - first, fourth - first)) / 6
