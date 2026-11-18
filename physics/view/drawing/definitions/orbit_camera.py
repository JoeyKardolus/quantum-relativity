"""Orbit camera: the view-projection matrix of a camera circling a centre.

Contract: `orbit_camera(centre, distance, yaw, pitch, aspect, fov_degrees=FOV_DEGREES)`
returns a (4, 4) float32 row-major matrix mapping world points to clip space (the
GL convention; upload transposed or column-major). The camera sits `distance`
from `centre`, turned `yaw` radians about the world y axis and `pitch` radians up,
looking at the centre with y up. Near and far planes are distance / 100 and
distance * 100. aspect is width / height, > 0.
"""
from __future__ import annotations

import numpy as np

from physics.view.drawing.definitions.camera_marks import FOV_DEGREES


def orbit_camera(centre: np.ndarray, distance: float, yaw: float, pitch: float, aspect: float,
                 fov_degrees: float = FOV_DEGREES) -> np.ndarray:
    """World to clip space for a camera orbiting `centre` at `distance`."""
    centre = np.asarray(centre, dtype=float)
    eye = centre + distance * np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch),
                                        np.cos(pitch) * np.cos(yaw)])
    forward = (centre - eye) / np.linalg.norm(centre - eye)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    near, far = distance / 100.0, distance * 100.0
    focal = 1.0 / np.tan(np.radians(fov_degrees) / 2.0)
    projection = np.zeros((4, 4))
    projection[0, 0] = focal / aspect
    projection[1, 1] = focal
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2.0 * far * near / (near - far)
    projection[3, 2] = -1.0
    return (projection @ view).astype(np.float32)
