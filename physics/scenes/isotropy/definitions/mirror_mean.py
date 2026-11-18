"""Mirror mean: a field's value at a place, averaged over every mirror and swap of that place.

Contract: `mirror_mean(field, centre, place)` takes a float array (across, across,
across) on a wrapping lattice, the centre's index along each axis (the same on all
three) and a place (i, j, k) counted from the centre. Returns the mean of the field
over every distinct place got by swapping the three entries and flipping their signs
(up to 48), a float. A lattice has no handedness left after that; only "on an axis" and
"off it" remain.
"""
from __future__ import annotations

import itertools

import numpy as np


def mirror_mean(field: np.ndarray, centre: int, place: tuple) -> float:
    """The mean of `field` over every mirror and swap of `place`, counted from `centre` (wrapping)."""
    across = field.shape[0]
    found = {tuple(sign * index for sign, index in zip(signs, order))
             for order in itertools.permutations(place) for signs in itertools.product((1, -1), repeat=3)}
    return float(np.mean([field[tuple((centre + index) % across for index in spot)] for spot in found]))
