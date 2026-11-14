"""Energy flow: energy moves between neighbouring energy volumes through their joints.

Contract: `energy_flow(grid, energy, rate)` returns the new energies. Each joint
(a, b) carries flux rate * (E_b - E_a) / (d_a + d_b), d the number of joints of a
volume, from b to a; a volume's change is the sum over its joints. Every flux
leaves one volume and enters the other, so the total is kept exactly (to
rounding); a volume's change is at most rate times the spread around it, stable
for rate <= 1.
"""
from __future__ import annotations

import numpy as np

from physics.cells.grid import Grid


def energy_flow(grid: Grid, energy: np.ndarray, rate: float) -> np.ndarray:
    """Energies after one round of symmetric flow through every joint."""
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    degree = np.bincount(grid.joints.ravel(), minlength=grid.cell_count)
    flux = rate * (energy[tails] - energy[heads]) / (degree[heads] + degree[tails])
    change = np.bincount(heads, flux, grid.cell_count) - np.bincount(tails, flux, grid.cell_count)
    return energy + change
