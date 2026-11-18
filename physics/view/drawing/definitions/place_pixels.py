"""Place pixels: how many pixels wide a lattice place is on screen.

Contract: `place_pixels(low, high, places, distance, height, fov_degrees=FOV_DEGREES)`
takes the grid's frame, its lattice places a side, the camera's distance from what it
looks at and the view's height in pixels, and returns the mean width of one lattice
place at that distance, in pixels. The outlines' opacity is read from it
(`outline_opacity`).
"""
from __future__ import annotations

import numpy as np

from physics.view.drawing.definitions.camera_marks import FOV_DEGREES


def place_pixels(low, high, places: int, distance: float, height: int, fov_degrees: float = FOV_DEGREES) -> float:
    """Pixels across one lattice place seen from `distance`."""
    place = float(np.max(np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64))) / max(int(places), 1)
    return place * height / (2.0 * max(float(distance), 1e-30) * np.tan(np.radians(fov_degrees) / 2.0))
