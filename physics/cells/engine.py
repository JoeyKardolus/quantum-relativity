"""Engine: the grid with the woven cells that are its exact, gap-free geometry.

Contract: a frozen record of
  grid          Grid (grid.py): energies, sizes, joints (thickness, rate) and seams; the
                rules and every measurement read only this
  corners       (k, 3) float: the corners of every cell (CONTEXT.md, map position): each
                energy volume has its own four, shared only with the joint cells on its
                faces; they describe the grid and decide nothing in it
  woven         CellComplex (cell_complex.py, cut by joint_cells.py): which corners make
                which cell; energy volumes, joint cells, seam cells and corner cells
  skeleton_corners  (s, 3) float: the skeleton's corners, the tetrahedral mesh cut once
                at the start (first_mesh.py) that says which cells neighbour which; each
                follows the cells around it (step.py) and is read only to place corners
                a collapse or split makes new (place_corners.py)
  collapses     joint collapses since the start
  splits        volume splits since the start
  start_space   the volume of all cells at the start, for size_vs_start
  start_mean_volume  the mean energy volume at the start; a volume past split_size
                times it splits (step.py)
  neighbour_changes  neighbour pairs that changed in a step without a collapse or a
                split; 0 by construction (step.py counts it to show it)
and, read through from woven for the app (mesh_message.py, frame_arrays.py):
  tets          (n, 4) int: each energy volume's own four corners, positively oriented
  faces         (f, 3) int: every triangle between two cells or on the outside, once,
                outward from its first cell
  face_cells    (f, 2) int: each triangle's first and second cell, -1 the outside of the
                open cube
  face_joint    (f,) int: the joint whose joint cell is a side of the triangle, else -1
  cell_kind     (c,) int: 0 energy volume, 1 joint cell, 2 seam cell, 3 corner cell
  skeleton_tets (n, 4) int: energy volume i's skeleton tetrahedron is row i
The topology changes only by joint collapse and volume split (step.py), on the
skeleton, after which the cells are cut again; otherwise a step moves corners only.
Made by new_engine(), advanced by step() (a pure function: Engine -> new Engine).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.cells.cell_complex import CellComplex
from physics.cells.grid import Grid


@dataclass(frozen=True)
class Engine:
    """Grid plus the woven cells (own corners per energy volume, joint cells between) and event counts."""

    grid: Grid
    corners: np.ndarray
    woven: CellComplex
    skeleton_corners: np.ndarray
    collapses: int
    splits: int
    start_space: float
    start_mean_volume: float
    neighbour_changes: int

    @property
    def tets(self) -> np.ndarray:
        """Each energy volume's own four corners."""
        return self.woven.tets

    @property
    def faces(self) -> np.ndarray:
        """Every triangle between two cells or on the outside."""
        return self.woven.faces

    @property
    def face_cells(self) -> np.ndarray:
        """Each triangle's two cells, -1 the outside."""
        return self.woven.face_cells

    @property
    def face_joint(self) -> np.ndarray:
        """Each triangle's joint, -1 when no joint cell is on it."""
        return self.woven.face_joint

    @property
    def cell_kind(self) -> np.ndarray:
        """Each cell's kind (cell_complex.py)."""
        return self.woven.cell_kind

    @property
    def skeleton_tets(self) -> np.ndarray:
        """Each energy volume's skeleton tetrahedron."""
        return self.woven.skeleton_tets
