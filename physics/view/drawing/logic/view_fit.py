"""View fit: where the camera should look so the grid fills most of the view.

Contract: `view_fit(low, high, fov_degrees=FOV_DEGREES, fitted_extent=None)` takes the grid's
bounding corners (grid lengths), the camera's field of view and the widest extent the
view was last fitted to (None: never fitted, or a fit is asked for by hand). It returns
(centre, distance, extent) when a fit is due, else None. A fit is due when
fitted_extent is None, or when the grid's extent has outgrown it by more than
REFIT_GROWTH (space grew past what the view was fitted to); shrinking never refits, so
the camera does not pump. centre is the bounds' middle (float (3,)), extent the widest
side (at least 1), distance FILL x extent / sin(fov / 2), so a side seen straight on
fills about half the view's height and the grid seen from a corner still fits.
"""
from __future__ import annotations

import numpy as np

from physics.view.drawing.definitions.camera_marks import FILL, FOV_DEGREES, REFIT_GROWTH


def view_fit(low, high, fov_degrees: float = FOV_DEGREES, fitted_extent: float | None = None):
    """The camera centre, distance and fitted extent when a fit is due, else None."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    extent = max(float(np.max(high - low)), 1.0)
    if fitted_extent is not None and extent <= REFIT_GROWTH * fitted_extent:
        return None
    centre = (low + high) / 2.0
    distance = FILL * extent / np.sin(np.radians(fov_degrees) / 2.0)
    return centre, float(distance), extent
