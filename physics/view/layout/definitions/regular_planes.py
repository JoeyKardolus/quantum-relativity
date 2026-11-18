"""Regular planes: where the corner planes of the perfect A B A B grid lie along one axis.

Contract: `regular_planes(count, first, volume, joint, places)` returns float64
(count + 1,): the positions of the corner planes `first` to `first + count` of a
lattice of `places` places whose even places are all `volume` wide and odd places all
`joint` wide, centred on 0. Plane 0 and plane `places` are the grid's outer frame.
"""
from __future__ import annotations

import numpy as np


def regular_planes(count: int, first: int, volume: float, joint: float, places: int) -> np.ndarray:
    """Positions of `count + 1` corner planes of the regular grid, from plane `first` on."""
    plane = first + np.arange(count + 1)
    return (plane // 2) * (volume + joint) + (plane % 2) * volume - 0.25 * places * (volume + joint)
