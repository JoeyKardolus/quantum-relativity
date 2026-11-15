"""Engine: the grid with the mesh that is its exact, gap-free cell geometry.

Contract: a frozen record of
  grid          Grid (grid.py): energies, sizes, joints (thickness, rate) and seams; the
                rules and every measurement read only this
  corners       (k, 3) float: the mesh's corners, map positions shared by every cell
                that meets there (CONTEXT.md, map position): they describe the grid and
                decide nothing in it
  tets          (n, 4) int: each energy volume's tetrahedron, its four corners,
                positively oriented (cell_volumes.py); the mesh's topology
  faces         (f, 3) int: each triangle's corner ring, outward from its first cell
                (tet_faces.py, derived from tets)
  face_cells    (f, 2) int: each face's first cell and second cell, -1 for the outside
                of the open cube; each cell has exactly 4 faces
  face_joint    (f,) int: the row of grid.joints each face is on, -1 outside; every
                face between two cells is exactly one joint
  collapses     joint collapses since the start (edges drawn into one corner)
  splits        volume splits since the start
  start_space   the mesh's volume at the start, for size_vs_start
  start_mean_volume  the mean energy volume at the start (start_space over the start's
                count); a volume past split_size times it splits (step.py)
  neighbour_changes  neighbour pairs that changed in a step without a collapse or a
                split; 0 by construction (step.py counts it to show it)
The topology (tets, faces, face_cells, face_joint) changes only by joint collapse and
volume split (step.py), by moves that leave only tetrahedra; otherwise a step moves
corners only.
Made by new_engine(), advanced by step() (a pure function: Engine -> new Engine).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.cells.grid import Grid


@dataclass(frozen=True)
class Engine:
    """Grid plus the shared-corner tetrahedral mesh that is its geometry, and event counts."""

    grid: Grid
    corners: np.ndarray
    tets: np.ndarray
    faces: np.ndarray
    face_cells: np.ndarray
    face_joint: np.ndarray
    collapses: int
    splits: int
    start_space: float
    start_mean_volume: float
    neighbour_changes: int
