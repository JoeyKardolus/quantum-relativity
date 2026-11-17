"""Grid cases (test helper): plain-array lattices for the layout tests, no engine.

`uneven_lattice(side, seed, spread)` returns (kind, length, packing) for a perfect
A B A B lattice whose volumes are about 10 long and joints about 1, each times
exp(spread * a standard normal). `inside(points, quads)` counts, for every 2D point,
the four-cornered faces (n, 4, 2) it lies strictly inside. `slice_faces(pieces, layer)`
returns the slice's faces (n, 4, 2) in drawing order with their cell ids.
"""
from __future__ import annotations

import numpy as np


def uneven_lattice(side: int, seed: int = 1, spread: float = 0.3) -> tuple:
    index = np.indices((side, side, side)).reshape(3, -1)
    kind = np.where((index % 2).sum(axis=0) == 0, 1, -1).astype(np.int8)
    noise = np.exp(spread * np.random.default_rng(seed).standard_normal(side ** 3))
    length = np.where(kind == 1, 10.0, 1.0) * noise
    return kind, length, (np.where(kind == 1, 10.0, 1.0) / length) ** 3


def slice_faces(pieces, layer: int) -> tuple:
    keep = pieces.place[:, 2] == layer
    corners = pieces.corners[keep].astype(np.float64)
    return (0.5 * (corners[:, :4] + corners[:, 4:]))[:, (0, 1, 3, 2), :2], pieces.cell[keep]


def inside(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    count = np.zeros(len(points), dtype=np.int64)
    for quad in quads:
        turns = []
        for one in range(4):
            start, end = quad[one], quad[(one + 1) % 4]
            turns.append((end[0] - start[0]) * (points[:, 1] - start[1]) - (end[1] - start[1]) * (points[:, 0] - start[0]))
        turns = np.array(turns)
        count += (turns > 0).all(axis=0) | (turns < 0).all(axis=0)
    return count
