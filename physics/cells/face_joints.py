"""Face joints: the joints of a tetrahedral mesh, one per triangle between two cells.

Contract: `face_joints(face_cells)` returns (joints, face_joint):
  joints      (k, 2) int, a < b, sorted unique: every pair of cells sharing a face
  face_joint  (f,) the row of joints each face is on, -1 for a face on the outside
Two tetrahedra share at most one triangle, so each joint has exactly one face.
Raises ValueError when two faces have the same two cells (not a simplicial mesh).
"""
from __future__ import annotations

import numpy as np


def face_joints(face_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joints from the faces' cell pairs and each face's joint row."""
    inside = np.flatnonzero((face_cells >= 0).all(axis=1))
    ends = np.sort(face_cells[inside], axis=1)
    joints, slot = np.unique(ends, axis=0, return_inverse=True)
    if len(joints) != len(inside):
        raise ValueError("two faces have the same two cells")
    face_joint = np.full(len(face_cells), -1, dtype=np.int64)
    face_joint[inside] = slot.ravel()
    return joints.astype(np.int64).reshape(-1, 2), face_joint
