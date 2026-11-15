"""Relax corners: move the shared corners so each cell's measured volume follows its target.

Contract: `relax_corners(corners, fan, targets, passes)` returns new corners (k, 3);
the topology (fan) is untouched, so no two cells start or stop neighbouring here.
Each pass is one least-squares Newton step on all cells at once: with e the cells'
errors target - V (cell_volumes.py), each clipped to MAX_CHANGE (5%) of the cell's
volume (of its target when the volume is not positive), and G the gradients of
the cells' volumes over the corners (per fan triangle (a, b, c): b x c / 6,
c x a / 6, a x b / 6, signed by the face side), the corners move by the smallest
displacement that fixes every clipped error to first order:
  dx = G^T y,   (G G^T + RIDGE * mean diagonal * I) y = e
(solved by conjugate gradients). A corner inside the mesh is shared by cells on all
sides, so a squeeze inside spreads outward through every cell to the outside
corners, which nothing pushes back: the open cube grows or shrinks at its edge.
The clip keeps each step inside the range where the straight-line guess holds, so
a cell far from its target (just split or collapsed) gets there over several
steps; a small cell is not crushed by big neighbours (every cell's error counts).
RIDGE (1e-9) only keeps the system definite. Raises RuntimeError if the solve
does not converge.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from physics.cells.cell_volumes import cell_volumes
from physics.cells.fan import Fan

MAX_CHANGE = 0.05
RIDGE = 1e-9


def relax_corners(corners: np.ndarray, fan: Fan, targets: np.ndarray, passes: int) -> np.ndarray:
    """Corners after `passes` least-squares Newton steps of every cell toward its target volume."""
    moved = np.array(corners, dtype=float)
    slot_count = len(fan.slot_cell)
    columns = (fan.slot_corner[:, None] * 3 + np.arange(3)).ravel()
    rows = np.repeat(fan.slot_cell, 3)
    for _ in range(passes):
        volume = cell_volumes(moved, fan)
        limit = MAX_CHANGE * np.where(volume > 0, volume, np.abs(targets))
        error = np.clip(targets - volume, -limit, limit)
        gradient = sp.csr_matrix((_slot_gradient(moved, fan, slot_count).ravel(), (rows, columns)),
                                 shape=(fan.cell_count, 3 * len(moved)))
        system = (gradient @ gradient.T).tocsr()
        diagonal = system.diagonal()
        ridge = RIDGE * float(diagonal.mean()) if len(diagonal) else 0.0
        system = system + sp.identity(fan.cell_count, format="csr") * ridge
        weight, failed = cg(system, error, rtol=1e-10, maxiter=20 * fan.cell_count,
                            M=sp.diags(1.0 / np.maximum(diagonal + ridge, 1e-300)))
        if failed:
            raise RuntimeError("corner relaxation did not converge")
        moved += (gradient.T @ weight).reshape(-1, 3)
    return moved


def _slot_gradient(corners: np.ndarray, fan: Fan, slot_count: int) -> np.ndarray:
    """Per (cell, corner) slot the gradient of that cell's volume over that corner."""
    centred = corners - corners.mean(axis=0)
    first, second, third = (centred[fan.triangles[:, column]] for column in range(3))
    per_vertex = np.stack([np.cross(second, third), np.cross(third, first),
                           np.cross(first, second)], axis=1).reshape(-1, 3) / 6
    signed = per_vertex[fan.entry_vertex] * fan.entry_sign[:, None]
    return np.stack([np.bincount(fan.entry_slot, signed[:, axis], slot_count)
                     for axis in range(3)], axis=1)
