"""Display layout of the energy volumes: neighbours sit at the sum of their radii.

Contract: `cell_layout(positions, joints, radius, iterations)` returns new
(n, 3) display positions after `iterations` local stress updates (Jacobi
stress majorization, damped by DAMPING): each energy volume moves toward the
mean of the places its neighbours want it, a neighbour wanting it at distance
radius_i + radius_j along their current direction. A joint longer than
WRAP_FACTOR times that distance is left out: the grid wraps around (a periodic
box), so a joint across the wrap cannot be drawn short without crumpling the
picture. A volume with no joint left in stays where it is. Display only; the
grid never sees these positions.
"""
from __future__ import annotations

import numpy as np

WRAP_FACTOR = 3.0
DAMPING = 0.5


def cell_layout(positions: np.ndarray, joints: np.ndarray, radius: np.ndarray,
                iterations: int) -> np.ndarray:
    """Positions relaxed so neighbouring volumes touch, wrap-around joints ignored."""
    count = len(positions)
    target = radius[joints[:, 0]] + radius[joints[:, 1]]
    for _ in range(iterations):
        offset = positions[joints[:, 0]] - positions[joints[:, 1]]
        distance = np.linalg.norm(offset, axis=1)
        near = distance <= WRAP_FACTOR * target
        if not near.any():
            break
        heads, tails = joints[near, 0], joints[near, 1]
        reach = offset[near] * (target[near] / np.maximum(distance[near], 1e-12))[:, None]
        cells = np.concatenate([heads, tails])
        wanted = np.concatenate([positions[tails] + reach, positions[heads] - reach])
        degree = np.bincount(cells, minlength=count)
        summed = np.stack([np.bincount(cells, wanted[:, axis], count) for axis in range(3)], axis=1)
        moved = degree > 0
        positions = positions.copy()
        positions[moved] = ((1 - DAMPING) * positions[moved]
                            + DAMPING * summed[moved] / degree[moved, None])
    return positions
