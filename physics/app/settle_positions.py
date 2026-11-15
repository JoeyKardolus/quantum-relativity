"""The drawing layout: touching volumes sit at the sum of their radii apart.

Contract: `settle_positions(positions, contacts, radii, iterations)` returns new display
positions (float64, (n, 3)) after `iterations` rounds of a simple averaged spring layout,
warm-started from `positions`: each round, every touching pair (a row of `contacts`, two
volume indices) is pulled or pushed along the line between them toward the distance
`radii[a] + radii[b]`, each side taking half; a volume moves by the mean of its pairs'
corrections times RELAX. Two volumes at the same point are first pulled apart along a
fixed direction, so the result never holds NaN. A volume without contacts stays where it
is. The input is not changed. Positions are for drawing only; nothing here feeds the physics.
"""
from __future__ import annotations

import numpy as np

RELAX = 0.8
_APART = np.array([0.57735027, 0.57735027, 0.57735027])


def settle_positions(positions, contacts, radii, iterations: int) -> np.ndarray:
    """Display positions after `iterations` rounds of the sum-of-radii layout."""
    settled = np.array(positions, dtype=float).reshape(-1, 3)
    pairs = np.asarray(contacts, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0 or len(settled) == 0:
        return settled
    radius = np.asarray(radii, dtype=float)
    first, second = pairs[:, 0], pairs[:, 1]
    target = radius[first] + radius[second]
    degree = np.bincount(pairs.ravel(), minlength=len(settled)).astype(float)
    share = np.where(degree > 0, RELAX / np.maximum(degree, 1.0), 0.0)[:, None]
    for _ in range(iterations):
        between = settled[second] - settled[first]
        length = np.linalg.norm(between, axis=1)
        together = length < 1e-12
        between[together] = _APART * np.maximum(target[together], 1e-9)[:, None]
        length[together] = np.linalg.norm(between[together], axis=1)
        correction = between * (0.5 * (length - target) / length)[:, None]
        moved = np.zeros_like(settled)
        for axis in range(3):
            moved[:, axis] += np.bincount(first, weights=correction[:, axis], minlength=len(settled))
            moved[:, axis] -= np.bincount(second, weights=correction[:, axis], minlength=len(settled))
        settled += moved * share
    return settled
