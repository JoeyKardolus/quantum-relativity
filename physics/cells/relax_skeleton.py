"""Relax skeleton: move the skeleton's shared corners so each tetrahedron's measured volume follows its target.

Contract: `relax_skeleton(corners, tets, targets, passes)` returns new corners (k, 3);
the topology (tets) is untouched, so no two cells start or stop neighbouring here.
Each pass is one regularised least-squares Gauss-Newton step on all tetrahedra at
once: with e the errors target - V (cell_volumes.py), each clipped to MAX_CHANGE (5%)
of the tetrahedron's volume (of its target when the volume is not positive), and G
the gradients of the volumes over the corners (for (a, b, c, d):
(c - a) x (d - a) / 6 at b, (d - a) x (b - a) / 6 at c, (b - a) x (c - a) / 6 at d,
minus their sum at a), each row of both divided by the tetrahedron's volume (so
every tetrahedron's error counts as a share of its size, and a small one is not
crushed by big neighbours), the corners move by
  dx = (G^T G + RIDGE * max(diag(G^T G), median diagonal))^-1 G^T e
(a sparse direct solve). A tetrahedral mesh has about six tetrahedra per corner, so
there are more volumes than corner directions: the fit is least squares and a
tetrahedron's size reaches its target only as far as its neighbours' let it.
RIDGE (1e-2, Levenberg-Marquardt damping, per corner direction by its own diagonal but
never below the median one, so a tiny packed tetrahedron damps only its own corners and
does not freeze the rest, while a weakly held corner still moves little) keeps the
system definite and the move small in the directions the volumes hold only weakly
(it holds the mesh's free translations and turns still). A corner inside is shared by tetrahedra on all sides, so a squeeze
inside spreads outward to the outside corners, which nothing pushes back: the open
cube grows or shrinks at its edge. A pass whose move would leave any tetrahedron
with less than KEEP_SHARE (1/2) of its volume is halved, up to HALVINGS (30) times,
so no tetrahedron is flipped or flattened by the relaxation. Raises RuntimeError if
the solve gives a value that is not finite or the halvings run out. The skeleton carries the large moves of a step; the cells' own
corners ride on it (step.py) and then settle their fine shape
(relax_corners.py), so a thin joint cell is never asked to follow a move far larger
than itself in one linear step.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from physics.cells.cell_volumes import cell_volumes

MAX_CHANGE = 0.05
RIDGE = 1e-2
KEEP_SHARE = 0.5
HALVINGS = 30


def relax_skeleton(corners: np.ndarray, tets: np.ndarray, targets: np.ndarray, passes: int) -> np.ndarray:
    """Corners after `passes` least-squares steps of every tetrahedron toward its target volume."""
    moved = np.array(corners, dtype=float)
    rows = np.repeat(np.arange(len(tets)), 12)
    columns = (tets[:, :, None] * 3 + np.arange(3)).reshape(-1)
    for _ in range(passes):
        volume = cell_volumes(moved, tets)
        limit = MAX_CHANGE * np.where(volume > 0, volume, np.abs(targets))
        scale = 1.0 / np.where(volume > 0, volume, np.abs(targets))
        error = np.clip(targets - volume, -limit, limit) * scale
        gradient = sp.csr_matrix(((_gradients(moved, tets) * scale[:, None, None]).ravel(),
                                  (rows, columns)), shape=(len(tets), 3 * len(moved)))
        system = (gradient.T @ gradient).tocsr()
        diagonal = system.diagonal()
        floor = float(np.median(diagonal)) if len(diagonal) else 0.0
        system = system + sp.diags(RIDGE * np.maximum(diagonal, floor))
        step = spsolve(system.tocsc(), gradient.T @ error)
        if not np.isfinite(step).all():
            raise RuntimeError("skeleton relaxation: the solve gave a value that is not finite")
        moved = _kept_shape(moved, step.reshape(-1, 3), tets, volume)
    return moved


def _gradients(corners: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Per tetrahedron (n, 4, 3) the gradient of its volume over each of its corners."""
    first, second, third, fourth = (corners[tets[:, column]] for column in range(4))
    at_second = np.cross(third - first, fourth - first) / 6
    at_third = np.cross(fourth - first, second - first) / 6
    at_fourth = np.cross(second - first, third - first) / 6
    at_first = -(at_second + at_third + at_fourth)
    return np.stack([at_first, at_second, at_third, at_fourth], axis=1)


def _kept_shape(corners: np.ndarray, step: np.ndarray, tets: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Corners moved by step, halved until every tetrahedron keeps KEEP_SHARE of its volume."""
    floor = KEEP_SHARE * volume
    for _ in range(HALVINGS):
        moved = corners + step
        if (cell_volumes(moved, tets) > floor).all():
            return moved
        step = step / 2
    raise RuntimeError("skeleton relaxation: no move keeps every tetrahedron the right way out")
