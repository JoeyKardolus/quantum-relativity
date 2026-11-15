"""Box energies: the A- and B-energy per box of a grid, on the host, as floats.

Contract: `box_energies(grid)` returns (a_energy, b_energy), two float64 numpy arrays of shape
(side, side, side): the grid's A- and B-energy per box in energy units (quanta times
QUANTUM, negatives read as 0). Copies both arrays from the device once; the grid is
not changed.
"""
from __future__ import annotations

import numpy as np

from physics.fluid import QUANTUM, Grid


def box_energies(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """(A-energy, B-energy) per box as float64 host arrays."""
    a_energy = np.maximum(grid.a_energy.numpy(), 0).astype(np.float64) * QUANTUM
    b_energy = np.maximum(grid.b_energy.numpy(), 0).astype(np.float64) * QUANTUM
    return a_energy, b_energy
