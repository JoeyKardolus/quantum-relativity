"""Cell targets: the volume each drawn cell should enclose.

Contract: `cell_targets(volume, thickness, face_joint, corners, fan)` returns (n,) per
cell its energy volume's own size (volume, the rest_volume of its energy, rule 4)
plus half of every joint on its faces: per face on joint j, area * thickness_j / 2
(face_areas.py). A cell is drawn as its energy volume with half of each wall around
it; the other half belongs to the neighbour. Faces on the outside carry no joint.
"""
from __future__ import annotations

import numpy as np

from physics.cells.face_areas import face_areas
from physics.cells.fan import Fan


def cell_targets(volume: np.ndarray, thickness: np.ndarray, face_joint: np.ndarray,
                 corners: np.ndarray, fan: Fan) -> np.ndarray:
    """Energy volume plus half of each joint wall on its faces, per cell."""
    inside = face_joint >= 0
    half = np.zeros(len(face_joint))
    half[inside] = face_areas(corners, fan)[inside] * thickness[face_joint[inside]] / 2
    heads, tails = fan.face_cells[inside, 0], fan.face_cells[inside, 1]
    return (volume + np.bincount(heads, half[inside], fan.cell_count)
            + np.bincount(tails, half[inside], fan.cell_count))
