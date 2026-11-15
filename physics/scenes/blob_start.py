"""Blob start: a grid of even balls of energy in empty space, each with its own velocity.

Contract: `blob_start(side, blobs, device="cuda")` returns a physics.fluid Grid of side**3
boxes, empty except for the given blobs. Each blob is a dict with `centre` (3 floats,
box units, box i's centre at i), `radius` (boxes; a box is in the ball when its centre
is within it), `energy` (the blob's total A-energy, spread evenly over its boxes) and
`velocity` (3 floats, boxes per time unit; A and B both move with it). Every box holds
B-energy new_grid.JOINT_SHARE times its A, as the uneven start does. Overlapping blobs
add their energies; the later blob's velocity wins in shared boxes. Raises ValueError
for a blob that covers no box or has energy <= 0.
"""
from __future__ import annotations

import numpy as np

from physics.fluid import Grid
from physics.fluid.new_grid import JOINT_SHARE


def blob_start(side: int, blobs: list[dict], device: str = "cuda") -> Grid:
    """Empty space holding even balls of A (and their joint share of B), each moving."""
    axis = np.arange(side, dtype=np.float64)
    a_energy = np.zeros((side, side, side))
    velocity = np.zeros((side, side, side, 3))
    for blob in blobs:
        centre = np.asarray(blob["centre"], dtype=np.float64)
        inside = ((axis[:, None, None] - centre[0]) ** 2 + (axis[None, :, None] - centre[1]) ** 2
                  + (axis[None, None, :] - centre[2]) ** 2) <= blob["radius"] ** 2
        if not inside.any() or blob["energy"] <= 0:
            raise ValueError(f"blob {blob} covers no box or holds no energy")
        a_energy[inside] += blob["energy"] / inside.sum()
        velocity[inside] = np.asarray(blob["velocity"], dtype=np.float64)
    return Grid.from_host(a_energy, JOINT_SHARE * a_energy, velocity, velocity, device=device)
