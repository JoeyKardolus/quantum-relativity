"""Lump centre: where a lump's extra energy sits, and how much of it there is.

Contract: `lump_centre(energy, near, radius)` takes the volumes' energy as a field
(n, n, n), a place `near` (three numbers, in volumes) and a radius in volumes. The extra
is what a volume holds above the field's median (the even fill), never below 0. Returns
`(centre, amount)`: the mean place of the extra within `radius` of `near`, the short
way round, as three floats not wrapped (so a path can be followed past the edge), and
the summed extra there. With no extra the centre is `near` and the amount 0.
"""
from __future__ import annotations

import numpy as np


def lump_centre(energy: np.ndarray, near, radius: float) -> tuple:
    """The centre of the extra energy within `radius` volumes of `near`, and its amount."""
    wide = energy.shape[0]
    extra = np.maximum(energy - np.median(energy), 0.0)
    places = np.indices(energy.shape).astype(np.float64)
    away = [(places[axis] - near[axis] + wide / 2.0) % wide - wide / 2.0 for axis in range(3)]
    extra = extra * (away[0] ** 2 + away[1] ** 2 + away[2] ** 2 <= radius ** 2)
    amount = float(extra.sum())
    if amount <= 0.0:
        return tuple(float(one) for one in near), 0.0
    return tuple(float(near[axis] + (extra * away[axis]).sum() / amount) for axis in range(3)), amount
