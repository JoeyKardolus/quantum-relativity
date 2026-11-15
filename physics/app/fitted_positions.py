"""The building positions brought to the drawing's scale.

Contract: `fitted_positions(positions, contacts, radii)` returns `positions` (float64,
(n, 3)) scaled about their mean so that the mean length of the touching pairs equals the
mean of `radii[a] + radii[b]` over them: new_state's scatter is in its own units, the
layout in radii, so the first frames do not jump. Unchanged (as float) when there are no
contacts or the pairs have no length. Display only.
"""
from __future__ import annotations

import numpy as np


def fitted_positions(positions, contacts, radii) -> np.ndarray:
    """The scatter positions rescaled so touching pairs sit about their summed radii apart."""
    points = np.array(positions, dtype=float).reshape(-1, 3)
    pairs = np.asarray(contacts, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        return points
    radius = np.asarray(radii, dtype=float)
    length = np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1).mean()
    target = (radius[pairs[:, 0]] + radius[pairs[:, 1]]).mean()
    if not (length > 0 and target > 0):
        return points
    centre = points.mean(axis=0)
    return centre + (points - centre) * (target / length)
