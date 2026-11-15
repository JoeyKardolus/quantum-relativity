"""View fit: where the camera should look so the occupied boxes fill about half the view.

Contract: `view_fit(low, high, fov_degrees, fitted_extent=None)` takes the occupied
boxes' bounding corners (box units), the camera's field of view and the widest
extent the view was last fitted to (None: never fitted, or a fit is asked for by
hand). It returns (centre, distance, extent) when a fit is due, else None. A fit
is due when fitted_extent is None, or when the occupied extent has outgrown it by
more than REFIT_GROWTH (space grew past what the view was fitted to); shrinking
never refits, so the camera does not pump. centre is the bounds' middle (float
(3,)), extent the widest side (at least 1 box), distance FILL x extent / sin(fov / 2),
so the extent fills about half the view and space has room to grow into.
"""
from __future__ import annotations

import numpy as np

FILL = 1.1
REFIT_GROWTH = 1.5


def view_fit(low, high, fov_degrees: float, fitted_extent: float | None = None):
    """The camera centre, distance and fitted extent when a fit is due, else None."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    extent = max(float(np.max(high - low)), 1.0)
    if fitted_extent is not None and extent <= REFIT_GROWTH * fitted_extent:
        return None
    centre = (low + high) / 2.0
    distance = FILL * extent / np.sin(np.radians(fov_degrees) / 2.0)
    return centre, float(distance), extent
