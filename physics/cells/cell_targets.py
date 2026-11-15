"""Cell targets: the volume each drawn tetrahedron should enclose.

Contract: `cell_targets(volume, thickness, faces, face_cells, face_joint, corners)`
returns (n,) per tetrahedron its energy volume's own size (volume, rest_volume of
its energy and quanta, rule 4) plus half of every joint on its faces: per face on
joint j, area * thickness_j / 2 (face_areas.py). A cell is drawn as its energy volume
with half of each wall around it; the other half belongs to the neighbour. Faces on
the outside carry no joint.
"""
from __future__ import annotations

import numpy as np

from physics.cells.face_areas import face_areas


def cell_targets(volume: np.ndarray, thickness: np.ndarray, faces: np.ndarray,
                 face_cells: np.ndarray, face_joint: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Energy volume plus half of each joint wall on its faces, per tetrahedron."""
    inside = face_joint >= 0
    half = face_areas(corners, faces[inside]) * thickness[face_joint[inside]] / 2
    count = len(volume)
    return (volume + np.bincount(face_cells[inside, 0], half, count)
            + np.bincount(face_cells[inside, 1], half, count))
