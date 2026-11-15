"""New grid: a ball of slightly uneven energy in the middle of a larger empty grid.

Contract: `new_grid(side, seed, unevenness, device="cuda")` returns a Grid (grid.py) of
side**3 boxes. Boxes whose centre lies within side / 4 of the grid's centre hold
energy, the rest are empty open space the ball can grow into. Per box in the ball,
with u1, u2 uniform in [-1, 1] (numpy default_rng(seed)), independently per box:
A-energy 1 + unevenness * u1 and B-energy JOINT_SHARE * (1 + unevenness * u2).
Everything starts at rest. The edge of the grid is a closed wall (step.py), so the
empty margin is what the ball has to grow before it meets it. Raises ValueError for
side < 8 or unevenness outside [0, 1) (energy must stay positive).
"""
from __future__ import annotations

import numpy as np

from physics.fluid.grid import Grid

JOINT_SHARE = 0.03


def new_grid(side: int, seed: int, unevenness: float, device: str = "cuda") -> Grid:
    """A seeded uneven ball of A- and B-energy at rest, radius side / 4, in empty space."""
    if side < 8:
        raise ValueError(f"side must be at least 8, got {side}")
    if not 0 <= unevenness < 1:
        raise ValueError(f"unevenness must be in [0, 1), got {unevenness}")
    generator = np.random.default_rng(seed)
    centre = (side - 1) / 2.0
    axis = np.arange(side) - centre
    distance = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)
    inside = distance <= side / 4.0
    a_noise = generator.uniform(-1.0, 1.0, size=inside.shape)
    b_noise = generator.uniform(-1.0, 1.0, size=inside.shape)
    a = np.where(inside, 1.0 + unevenness * a_noise, 0.0)
    b = np.where(inside, JOINT_SHARE * (1.0 + unevenness * b_noise), 0.0)
    return Grid.from_host(a, b, device=device)
