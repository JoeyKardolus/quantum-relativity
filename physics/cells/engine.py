"""Engine: the grid with the scaffolding that gives it exact, gap-free cell shapes.

Contract: a frozen record of
  grid          Grid (grid.py): energies, sizes, joints and joint volumes; the rules and
                every measurement read only this. Its joints are the woven neighbour
                graph: the shared faces of the first tessellation (new_engine.py),
                changed since only by joint collapse (step.py).
  positions     (n, 3) a working position per energy volume in [0, side)^3, a label
                for computing shapes only (bible rule 2)
  weights       (n,) a power weight per energy volume, a label like the positions:
                fitted each step so each cell's tessellated volume follows its target
                size (fit_weights.py)
  side          side of the periodic cube; side**3 = grid.total_space() (energy
                volumes plus joints fill all of space)
  tessellation  Tessellation of positions and weights in that cube: the shapes the
                viewer draws (shapes.py); its faces are the grid's joints except
                where the shapes changed a pair by themselves (neighbour_changes)
  collapses     energy volumes that fell into others since the start
  start_space   grid.total_space() at the start, for size_vs_start
  neighbour_changes  neighbour pairs the shapes changed by themselves, outside a
                joint collapse, since the start (neighbour_changes.py); the grid
                does not follow them
Made by new_engine(), advanced by step() (a pure function: Engine -> new Engine).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.cells.grid import Grid
from physics.cells.tessellation import Tessellation


@dataclass(frozen=True)
class Engine:
    """Grid plus working positions, weights and box, the tessellation they give, and counts."""

    grid: Grid
    positions: np.ndarray
    weights: np.ndarray
    side: float
    tessellation: Tessellation
    collapses: int
    start_space: float
    neighbour_changes: int
