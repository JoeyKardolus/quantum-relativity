"""Relax corners: move every corner so each cell's measured size follows its target.

Contract: `relax_corners(corners, faces, face_cells, targets, passes)` returns new corners
(k, 3); the topology (faces, face_cells) is untouched, so no two cells start or stop
neighbouring here. Every corner moves: the energy volumes' own corners and the joint
cells', seam cells' and corner cells' alike; nothing holds a cell to one shape (rule 4),
only its size is asked for. Each pass is one damped Gauss-Newton step on all cells at
once: with e the errors target - V (cell_sizes.py), each clipped to MAX_CHANGE (5%) of
the cell's size (of its target when the size is not positive), and G the gradients of
the sizes over the corners (per triangle (a, b, c) turned outward from the cell,
(b - r) x (c - r) / 6 at a and its turns at b and c), each row of both divided by the
cell's size (so every cell's error counts as a share of its size, and a thin joint
cell is not crushed by big energy volumes), the corners move by the least move that
meets the errors,
  dx = G^T (G G^T + RIDGE * diag(G G^T))^-1 e
(conjugate gradients over the cells, Jacobi preconditioned, to a relative residual of
SOLVE_TOLERANCE (1e-8), the system being symmetric positive definite). There are about
three corners per cell, far more directions than sizes, so every size can be met and
the rest of the shape is left as it is. RIDGE (1e-2, Levenberg-Marquardt damping per
cell) keeps the system definite. The cells differ in size by many powers of ten (a
packed energy volume's corner cells are tiny), so no corner moves further in one pass
than REACH (0.25) of the smallest cell it is on, measured as 3 V / (its surface area),
the inradius of a tetrahedron and about a joint cell's half thickness: a move is only
asked of a cell where its linear step holds. A corner inside is shared by cells on all sides, so a squeeze inside spreads outward
to the outside corners, which nothing pushes back: the open cube grows or shrinks at
its edge. A pass whose move would leave any cell with less than KEEP_SHARE (1/2) of
its size is halved, up to HALVINGS (30) times, so no cell is flipped or flattened by
the relaxation. Raises RuntimeError if the solve does not converge or gives a value
that is not finite, or the halvings run out.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from physics.cells.cell_sizes import cell_sizes

MAX_CHANGE = 0.05
RIDGE = 1e-2
KEEP_SHARE = 0.5
HALVINGS = 30
REACH = 0.25
SOLVE_TOLERANCE = 1e-8


def relax_corners(corners: np.ndarray, faces: np.ndarray, face_cells: np.ndarray, targets: np.ndarray,
                  passes: int) -> np.ndarray:
    """Corners after `passes` least-move steps of every cell toward its target size."""
    moved = np.array(corners, dtype=float)
    count = len(targets)
    cell, ring = _sides(faces, face_cells)
    anchor = np.zeros(count, dtype=np.int64)
    anchor[cell[::-1]] = ring[::-1, 0]
    rows = np.repeat(cell, 9)
    columns = (ring[:, :, None] * 3 + np.arange(3)).reshape(-1)
    for _ in range(passes):
        size = cell_sizes(moved, faces, face_cells, count)
        base = np.where(size > 0, size, np.abs(targets))
        error = np.clip(targets - size, -MAX_CHANGE * base, MAX_CHANGE * base) / base
        values = (_gradients(moved, ring, anchor[cell]) / base[cell][:, None, None]).reshape(-1)
        gradient = sp.csr_matrix((values, (rows, columns)), shape=(count, 3 * len(moved)))
        system = (gradient @ gradient.T).tocsr()
        diagonal = (1 + RIDGE) * system.diagonal()
        system = system + sp.diags(RIDGE * system.diagonal())
        solved, failed = cg(system, error, rtol=SOLVE_TOLERANCE, M=sp.diags(1.0 / diagonal), maxiter=10 * count)
        if failed:
            raise RuntimeError("corner relaxation: the solve did not converge")
        step = (gradient.T @ solved).reshape(-1, 3)
        if not np.isfinite(step).all():
            raise RuntimeError("corner relaxation: the solve gave a value that is not finite")
        reach = REACH * _corner_scale(moved, ring, cell, size)
        length = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, np.divide(reach, length, out=np.ones_like(length), where=length > 0))[:, None]
        moved = _kept_shape(moved, step, faces, face_cells, size)
    return moved


def _sides(faces: np.ndarray, face_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every (cell, triangle turned outward from it) pair, the outside left out."""
    inner = face_cells[:, 1] >= 0
    return (np.concatenate([face_cells[:, 0], face_cells[inner, 1]]),
            np.concatenate([faces, faces[inner][:, ::-1]]))


def _corner_scale(corners: np.ndarray, ring: np.ndarray, cell: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Per corner the smallest 3 V / (surface area) of the cells it is on."""
    first, second, third = (corners[ring[:, column]] for column in range(3))
    area = np.bincount(cell, np.linalg.norm(np.cross(second - first, third - first), axis=1) / 2, len(size))
    scale = 3 * np.abs(size) / np.where(area > 0, area, 1.0)
    smallest = np.full(len(corners), np.inf)
    np.minimum.at(smallest, ring.ravel(), np.repeat(scale[cell], 3))
    return smallest


def _gradients(corners: np.ndarray, ring: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Per (cell, triangle) pair (m, 3, 3) the gradient of the cell's size over the triangle's corners."""
    reference = corners[anchor]
    first, second, third = (corners[ring[:, column]] - reference for column in range(3))
    return np.stack([np.cross(second, third), np.cross(third, first), np.cross(first, second)], axis=1) / 6


def _kept_shape(corners: np.ndarray, step: np.ndarray, faces: np.ndarray, face_cells: np.ndarray,
                size: np.ndarray) -> np.ndarray:
    """Corners moved by step, halved until every cell keeps KEEP_SHARE of its size."""
    floor = KEEP_SHARE * size
    for _ in range(HALVINGS):
        moved = corners + step
        if (cell_sizes(moved, faces, face_cells, len(size)) > floor).all():
            return moved
        step = step / 2
    raise RuntimeError("corner relaxation: no move keeps every cell the right way out")
