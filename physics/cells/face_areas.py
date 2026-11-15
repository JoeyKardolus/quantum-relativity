"""Face areas: each face's area, from the corners and the fan.

Contract: `face_areas(corners, fan)` returns (f,) per face the summed area of its
fan triangles, |(b - a) x (c - a)| / 2 each (a face need not be flat).
"""
from __future__ import annotations

import numpy as np

from physics.cells.fan import Fan


def face_areas(corners: np.ndarray, fan: Fan) -> np.ndarray:
    """Per face the area of its fan of triangles."""
    first, second, third = (corners[fan.triangles[:, column]] for column in range(3))
    area = np.linalg.norm(np.cross(second - first, third - first), axis=1) / 2
    return np.bincount(fan.tri_face, area, len(fan.face_cells))
