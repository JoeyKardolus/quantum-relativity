"""Far to near: the order to draw see-through cells in, the farthest from the eye first.

Contract: `far_to_near(centres, view_projection)` takes cell centres float (n, 3) and a
(4, 4) row-major world-to-clip matrix (`orbit_camera`) and returns uint32 (n,): the
cells' row numbers, the farthest along the view first. Blending one cell over another
is only right in this order. The distance is the clip-space w (the depth along the
view), cut into LEVELS equal steps between the farthest and the nearest centre so the
sort is a radix sort (a few milliseconds for a quarter of a million cells); cells in
the same step keep their row order.
"""
from __future__ import annotations

import numpy as np

from physics.view.drawing.definitions.camera_marks import LEVELS


def far_to_near(centres: np.ndarray, view_projection: np.ndarray) -> np.ndarray:
    """Row numbers of the cells, the farthest from the eye first."""
    matrix = np.asarray(view_projection, dtype=np.float32)
    depth = np.asarray(centres, dtype=np.float32) @ matrix[3, :3] + matrix[3, 3]
    if not len(depth):
        return np.zeros(0, dtype=np.uint32)
    farthest, nearest = float(depth.max()), float(depth.min())
    steps = ((farthest - depth) * (LEVELS / max(farthest - nearest, 1e-30))).astype(np.uint16)
    return np.argsort(steps, kind="stable").astype(np.uint32)
