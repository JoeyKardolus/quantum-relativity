"""Mesh seams: the seams of the woven cells, the seam cells where joint cells meet by their thin sides.

Contract: `mesh_seams(face_cells, face_joint, cell_kind)` returns (r, 2) int rows
(seam, joint): a seam is a seam cell (cell_complex.py) that shares faces with at least
two joint cells, and each row names one joint whose joint cell shares a face with it.
Seams are numbered 0 .. m-1 in the order of their seam cells (the skeleton's edges, in
order of their sorted corner pair); rows are sorted by seam, then joint. A joint cell
touches the seam cells of its face's three edges (fewer on the outside). The volumes
around a seam are the two energy volumes of each of its joints.
"""
from __future__ import annotations

import numpy as np

from physics.cells.cell_complex import KIND_SEAM


def mesh_seams(face_cells: np.ndarray, face_joint: np.ndarray, cell_kind: np.ndarray) -> np.ndarray:
    """Per seam cell touching two or more joint cells, one (seam, joint) row per such joint."""
    first = face_cells[:, 0]
    touching = (cell_kind[first] == KIND_SEAM) & (face_joint >= 0)
    pairs = np.unique(np.column_stack([first[touching], face_joint[touching]]), axis=0).reshape(-1, 2)
    cells, slot, count = np.unique(pairs[:, 0], return_inverse=True, return_counts=True)
    kept = count[slot.ravel()] >= 2
    seam_of_cell = np.cumsum(count >= 2) - 1
    return np.column_stack([seam_of_cell[slot.ravel()[kept]], pairs[kept, 1]]).astype(np.int64).reshape(-1, 2)
