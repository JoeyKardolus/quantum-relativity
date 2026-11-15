"""New grid: space filled with uneven energy, high and low zones at every size, at rest.

Contract: `new_grid(side, seed, unevenness, slope=SLOPE, device="cuda")` returns a
Grid (grid.py) of side**3 boxes. Every box at least `margin = side // 16` boxes from
the grid's outer faces holds energy (the filled cube, no empty surroundings inside
it); the margin outside is empty open space the outer region can grow into before it
meets the closed wall (step.py, stats edge_share). Inside the filled cube, with g the
uneven field (uneven_field.py: seed, slope, root mean square `unevenness`), a box
holds A-energy proportional to exp(g), scaled so the filled cube's mean A per box is
exactly 1 (log A follows g's spectrum: a log-normal field, so energy stays positive
at any unevenness), and B-energy JOINT_SHARE times its A-energy: a zone is high or
low in energy of both kinds, B its joint share, so B's compression equals A at the
start. `slope` is the spectral slope of g (P(k) = k**slope); the default -3 gives
every octave of wavelength the same share of the unevenness. Everything starts at
rest. Raises ValueError for side < 8 and the uneven field's errors.
"""
from __future__ import annotations

import numpy as np

from physics.fluid.grid import Grid
from physics.fluid.uneven_field import uneven_field

JOINT_SHARE = 0.03
SLOPE = -3.0


def new_grid(side: int, seed: int, unevenness: float, slope: float = SLOPE, device: str = "cuda") -> Grid:
    """A seeded energy-filled cube with ripples at every size, at rest, in an empty margin."""
    field = uneven_field(side, seed, slope, unevenness)
    margin = side // 16
    inside = np.zeros((side, side, side), dtype=bool)
    inside[margin:side - margin, margin:side - margin, margin:side - margin] = True
    level = np.exp(field - field[inside].max())
    a = np.where(inside, level / level[inside].mean(), 0.0)
    return Grid.from_host(a, JOINT_SHARE * a, device=device)
