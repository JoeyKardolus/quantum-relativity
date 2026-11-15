"""Engine: the grid with the mesh that is its exact, gap-free cell geometry.

Contract: a frozen record of
  grid          Grid (grid.py): energies, sizes, joints (thickness, rate) and seams; the
                rules and every measurement read only this
  corners       (k, 3) float: the mesh's corners, map positions shared by every cell
                that meets there (CONTEXT.md, map position): they describe the grid and
                decide nothing in it
  faces         tuple of int arrays: each face's corner ring, outward from its first cell
  face_cells    (f, 2) int: each face's first cell and second cell, -1 for the outside
                of the open cube; each cell is the closed set of its faces
  face_joint    (f,) int: the row of grid.joints each face is on, -1 outside; a
                joint's faces are exactly the faces between its two cells
  fan           Fan (fan.py): the faces as triangles, derived from the topology
  collapses     joint collapses since the start (energy volumes that fell into others)
  splits        volume splits since the start
  start_space   the mesh's volume at the start, for size_vs_start
  neighbour_changes  neighbour pairs that changed in a step without a collapse or a
                split; 0 by construction (step.py counts it to show it)
The topology (faces, face_cells, face_joint) changes only by joint collapse and
volume split (step.py); otherwise a step moves corners only.
Made by new_engine(), advanced by step() (a pure function: Engine -> new Engine).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.cells.fan import Fan
from physics.cells.grid import Grid


@dataclass(frozen=True)
class Engine:
    """Grid plus the shared-corner mesh that is its geometry, and event counts."""

    grid: Grid
    corners: np.ndarray
    faces: tuple
    face_cells: np.ndarray
    face_joint: np.ndarray
    fan: Fan
    collapses: int
    splits: int
    start_space: float
    neighbour_changes: int
