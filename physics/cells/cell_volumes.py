"""Cell volumes: each cell's measured volume, from the corners and the fan.

Contract: `cell_volumes(corners, fan)` returns (n,) per cell the volume its closed set
of faces encloses: per triangle (a, b, c) the signed volume det(a, b, c) / 6 about the
corners' mean, added to the face's first cell and taken from its second. Faces point
outward from their first cell, so a well-shaped cell has a positive volume; the
volumes sum to the volume of the whole mesh (its outside faces).
"""
from __future__ import annotations

import numpy as np

from physics.cells.fan import Fan


def cell_volumes(corners: np.ndarray, fan: Fan) -> np.ndarray:
    """Per cell the signed volume enclosed by its faces."""
    centred = corners - corners.mean(axis=0)
    first, second, third = (centred[fan.triangles[:, column]] for column in range(3))
    tetra = np.einsum("ij,ij->i", first, np.cross(second, third)) / 6
    face_volume = np.bincount(fan.tri_face, tetra, len(fan.face_cells))
    heads, tails = fan.face_cells[:, 0], fan.face_cells[:, 1]
    inside = tails >= 0
    return (np.bincount(heads, face_volume, fan.cell_count)
            - np.bincount(tails[inside], face_volume[inside], fan.cell_count))
