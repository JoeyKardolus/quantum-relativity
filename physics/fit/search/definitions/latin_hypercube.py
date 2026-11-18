"""Latin hypercube: the random phase's points, spread so every axis is covered evenly.

Contract: `latin_hypercube(points, axes, seed)` returns a float array of shape
(points, axes) in the unit cube. Along every axis the range is cut into `points` equal
strata and each stratum holds exactly one point, at a random place inside it; the axes
are shuffled apart from each other. The same seed gives the same points. The first so
many rows are a plain random sample of the cube. `points = 0` gives no rows.
"""
from __future__ import annotations

import numpy as np


def latin_hypercube(points: int, axes: int, seed: int) -> np.ndarray:
    """`points` points of the unit cube, one per stratum along each of `axes` axes."""
    generator = np.random.default_rng(seed)
    columns = [(generator.permutation(points) + generator.random(points)) / max(points, 1) for _ in range(axes)]
    return np.stack(columns, axis=1)
