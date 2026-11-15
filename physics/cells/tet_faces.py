"""Tet faces: the triangles of a tetrahedral mesh, each once, with the one or two cells on it.

Contract: `tet_faces(tets)` returns (faces, face_cells) for positively oriented
tetrahedra (n, 4):
  faces       (f, 3) int corner indices, each face once, sorted by its sorted corners;
              the ring is outward from its first cell
  face_cells  (f, 2) int: the face's first cell (the lower index of the two) and its
              second; -1 for a face on the outside (it borders nothing)
Each tetrahedron (a, b, c, d) has the outward faces (b, c, d), (a, d, c), (a, b, d),
(a, c, b). Raises ValueError when a triangle is on more than two tetrahedra (not a
mesh of space).
"""
from __future__ import annotations

import numpy as np

_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def tet_faces(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every triangle of the mesh once, outward from its first cell, and its cells (-1 outside)."""
    rings = tets[:, _FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(tets)), 4)
    keys = np.sort(rings, axis=1)
    order = np.lexsort((owner, keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, rings, owner = keys[order], rings[order], owner[order]
    starts = np.flatnonzero(np.concatenate([[True], (keys[1:] != keys[:-1]).any(axis=1)]))
    count = np.diff(np.append(starts, len(keys)))
    if (count > 2).any():
        raise ValueError("a triangle is on more than two tetrahedra")
    faces = rings[starts]
    face_cells = np.full((len(starts), 2), -1, dtype=np.int64)
    face_cells[:, 0] = owner[starts]
    pair = count == 2
    face_cells[pair, 1] = owner[starts[pair] + 1]
    return faces.astype(np.int64), face_cells
