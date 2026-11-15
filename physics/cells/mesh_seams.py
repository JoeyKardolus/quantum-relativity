"""Mesh seams: the seams of the mesh, the edges where joint walls meet.

Contract: `mesh_seams(faces, face_joint)` returns (r, 2) int rows (seam, joint): a seam
is a mesh edge that lies on at least two joint faces (an inner edge, with the ring of
tetrahedra around it, or an edge on the outside with at least three tetrahedra
around it), and each row names one joint whose face has that edge. Seams are
numbered 0 .. m-1 in order of their sorted corner pair; rows are sorted by seam,
then joint. A joint meets the seams of its face's three edges (fewer on the outside).
The volumes around a seam are the two cells of each of its joints.
"""
from __future__ import annotations

import numpy as np


def mesh_seams(faces: np.ndarray, face_joint: np.ndarray) -> np.ndarray:
    """Per edge on two or more joint faces, one (seam, joint) row per such joint."""
    inside = face_joint >= 0
    rings, joint = faces[inside], face_joint[inside]
    low = np.minimum(rings, np.roll(rings, -1, axis=1)).ravel()
    high = np.maximum(rings, np.roll(rings, -1, axis=1)).ravel()
    owner = np.repeat(joint, 3)
    width = int(faces.max()) + 1 if len(faces) else 1
    keys = low * width + high
    order = np.lexsort((owner, keys))
    keys, owner = keys[order], owner[order]
    unique_keys, slot, count = np.unique(keys, return_inverse=True, return_counts=True)
    seam_of_key = np.cumsum(count >= 2) - 1
    kept = count[slot] >= 2
    return np.column_stack([seam_of_key[slot[kept]], owner[kept]]).astype(np.int64).reshape(-1, 2)
