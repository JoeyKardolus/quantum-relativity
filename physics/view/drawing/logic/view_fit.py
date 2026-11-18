"""View fit: where the camera should look so the grid's frame fills the view.

Contract: `view_fit(low, high, fov_degrees=FOV_DEGREES, fill=FILL)` takes the corners of
the grid's outer frame (grid lengths) and the camera's field of view and returns
(centre, distance, extent): centre is the frame's middle (float (3,)), extent its widest
side (at least 1), distance fill x extent / sin(fov / 2). At FILL a side seen straight
on fills about half the view's height and the grid seen from a corner still fits; at
FACE_FILL the slice seen straight on fills nearly all of it. The frame is the regular
grid's, so it changes only when the typical cell does, and smoothly: a drawer that fits
every frame keeps the grid still on screen.
"""
from __future__ import annotations

import numpy as np

from physics.view.drawing.definitions.camera_marks import FILL, FOV_DEGREES


def view_fit(low, high, fov_degrees: float = FOV_DEGREES, fill: float = FILL) -> tuple:
    """The camera centre and distance that fit the frame, and the frame's widest side."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    extent = max(float(np.max(high - low)), 1.0)
    return (low + high) / 2.0, float(fill * extent / np.sin(np.radians(fov_degrees) / 2.0)), extent
