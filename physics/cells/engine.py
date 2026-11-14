"""Engine: the grid with the scaffolding that gives it exact, gap-free cell shapes.

Contract: a frozen record of
  grid          Grid (grid.py): energies, sizes, joints and joint volumes; the rules and
                every measurement read only this. Its joints are exactly the shared
                faces of `tessellation`, one joint per pair of neighbouring cells.
  positions     (n, 3) a working position per energy volume in [0, side)^3, a label
                for computing shapes only (bible rule 2)
  weights       (n,) a power weight per energy volume, a label like the positions:
                fitted each step so each cell's tessellated volume follows its target
                size (fit_weights.py)
  side          side of the periodic cube; side**3 = grid.total_space() (energy
                volumes plus joints fill all of space)
  tessellation  Tessellation of positions and weights in that cube: the shapes the
                viewer draws (shapes.py)
  collapses     energy volumes that fell into others since the start
  start_space   grid.total_space() at the start, for size_vs_start
Made by new_engine(), advanced by step() (a pure function: Engine -> new Engine).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.cells.grid import Grid
from physics.cells.tessellation import Tessellation


@dataclass(frozen=True)
class Engine:
    """Grid plus working positions, weights and box, and the tessellation they give."""

    grid: Grid
    positions: np.ndarray
    weights: np.ndarray
    side: float
    tessellation: Tessellation
    collapses: int
    start_space: float
