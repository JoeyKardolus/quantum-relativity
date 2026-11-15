"""Joint offsets: how far each energy volume's faces stand back from its skeleton tetrahedron's faces.

Contract: `joint_offsets(skeleton_corners, skeleton_tets, slot_joint, thickness, share)`
returns (n, 4): per skeleton tetrahedron and slot s, for the face opposite s,
  inner face   half the thickness of the joint on it (slot_joint >= 0): the joint cell
               starts with one half on each side
  outer face   share * V ** (1/3) / 2, V the skeleton tetrahedron's volume: half a rest
               thickness (joint_rest.py), so the corner and seam cells on the outside
               keep a size; nothing fills it (an outer face borders nothing)
each at most CAP (0.25) of the tetrahedron's inradius 3 V / (sum of its face areas), so
even a sliver keeps an energy volume the right way out. Used by place_corners.py.
"""
from __future__ import annotations

import numpy as np

from physics.cells.cell_volumes import cell_volumes

CAP = 0.25
_OPPOSITE = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def joint_offsets(skeleton_corners: np.ndarray, skeleton_tets: np.ndarray, slot_joint: np.ndarray,
                  thickness: np.ndarray, share: float) -> np.ndarray:
    """Per skeleton tetrahedron and slot, the offset of the energy volume's face opposite it."""
    volume = cell_volumes(skeleton_corners, skeleton_tets)
    areas = np.zeros((len(skeleton_tets), 4))
    for slot, ring in enumerate(_OPPOSITE):
        first, second, third = (skeleton_corners[skeleton_tets[:, column]] for column in ring)
        areas[:, slot] = np.linalg.norm(np.cross(second - first, third - first), axis=1) / 2
    inradius = 3 * volume / areas.sum(axis=1)
    outer = np.broadcast_to((share * np.cbrt(np.abs(volume)) / 2)[:, None], slot_joint.shape)
    halves = np.where(slot_joint >= 0, thickness[np.maximum(slot_joint, 0)] / 2 if len(thickness) else 0.0,
                      outer)
    return np.minimum(halves, CAP * inradius[:, None])
