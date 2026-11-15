"""Face areas: each triangle's area, from the corners.

Contract: `face_areas(corners, faces)` returns (f,) per triangle (a, b, c) of faces
(f, 3) its area |(b - a) x (c - a)| / 2.
"""
from __future__ import annotations

import numpy as np


def face_areas(corners: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per triangle its area."""
    first, second, third = (corners[faces[:, column]] for column in range(3))
    return np.linalg.norm(np.cross(second - first, third - first), axis=1) / 2
