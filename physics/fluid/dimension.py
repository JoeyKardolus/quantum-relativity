"""Dimension: how the measured volume within a distance grows, read as the space's dimension.

Contract: `dimension(distance, length, inner, outer)` takes numpy arrays per box of
proper distance from one origin (proper_distance.py) and local length
(local_length.py) and two radii 0 < inner < outer; it returns
log(V(outer) / V(inner)) / log(outer / inner), V(r) the summed length cubed of the
boxes within distance r: d where V grows as r**d (bible rule 1: dimension measured
from the grid alone, by neighbour counts and paths). Validated on the flat reference
(every box equal; tests). Raises ValueError for radii out of order or a ball holding
nothing.
"""
from __future__ import annotations

import math

import numpy as np


def dimension(distance: np.ndarray, length: np.ndarray, inner: float, outer: float) -> float:
    """The exponent of measured volume against proper radius between two radii."""
    if not 0 < inner < outer:
        raise ValueError(f"radii must satisfy 0 < inner < outer, got {inner}, {outer}")
    cubes = np.where(np.isfinite(distance), length, 0.0) ** 3
    small = float(cubes[distance <= inner].sum())
    large = float(cubes[distance <= outer].sum())
    if small <= 0.0:
        raise ValueError(f"no measured volume within {inner}")
    return math.log(large / small) / math.log(outer / inner)
