"""Build the starting grid from tetrahedra used as scaffolding, then drop the shapes.

Contract: `build_grid(cells, seed, unevenness)` returns (Grid, display_positions):
  scaffold   round(cells / 6.77) points scattered uniformly in a periodic unit box
             (a 3D Delaunay tetrahedralisation of random points has about 6.77
             tetrahedra per point), tiled 27 times and tetrahedralised with
             scipy.spatial.Delaunay; the tetrahedra whose centroid lies in the
             central box are the energy volumes, so the grid wraps with no
             boundary and every energy volume has exactly 4 joints.
  joints     one per pair of tetrahedra sharing a side (matched across the wrap).
  energy     1 + unevenness * u, u uniform in [-1, 1] per energy volume (seeded).
  volume     rest_volume(energy): the energy-volume rule, not the tetrahedron's size.
  squeeze    0 per joint; merged_into None.
display_positions are the tetrahedra centroids wrapped into the box, (n, 3),
for drawing only; the Grid never holds them. Raises ValueError when
unevenness is outside [0, 1) (energy must stay positive) or cells < 30.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import Delaunay

from physics.cells.grid import Grid
from physics.cells.rest_volume import rest_volume

_TETRAHEDRA_PER_POINT = 6.77
_OFFSETS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)


def build_grid(cells: int, seed: int, unevenness: float) -> tuple[Grid, np.ndarray]:
    """A periodic grid of about `cells` energy volumes with slightly uneven energy."""
    if not 0 <= unevenness < 1:
        raise ValueError("unevenness must be in [0, 1) so every energy stays positive")
    if cells < 30:
        raise ValueError("ask for at least 30 cells so the periodic scaffold is sound")
    rng = np.random.default_rng(seed)
    point_count = max(int(round(cells / _TETRAHEDRA_PER_POINT)), 5)
    points = rng.random((point_count, 3))
    tiled = (points[None, :, :] + _OFFSETS[:, None, :]).reshape(-1, 3)
    mesh = Delaunay(tiled)
    centroids = tiled[mesh.simplices].mean(axis=1)
    kept = np.flatnonzero(np.all((centroids >= 0) & (centroids < 1), axis=1))
    joints = _joints(mesh, kept, point_count)
    energy = 1.0 + unevenness * rng.uniform(-1.0, 1.0, len(kept))
    grid = Grid(cell_count=len(kept), joints=joints, energy=energy,
                volume=rest_volume(energy), squeeze=np.zeros(len(joints)))
    return grid, centroids[kept]


def _joints(mesh: Delaunay, kept: np.ndarray, point_count: int) -> np.ndarray:
    """Pairs of kept tetrahedra sharing a side, a neighbour across the wrap found by
    its translation-free key."""
    keys = _translation_free_keys(mesh.simplices, point_count)
    _, label = np.unique(keys, axis=0, return_inverse=True)
    label = label.ravel()
    cell_of_label = np.full(label.max() + 1, -1)
    cell_of_label[label[kept]] = np.arange(len(kept))
    neighbours = mesh.neighbors[kept]
    if (neighbours < 0).any():
        raise ValueError("scaffold reached the hull: too few points for a periodic grid")
    far = cell_of_label[label[neighbours]]
    near = np.repeat(np.arange(len(kept)), 4)
    pairs = np.sort(np.column_stack([near, far.ravel()]), axis=1)
    return np.unique(pairs, axis=0).astype(np.int64)


def _translation_free_keys(simplices: np.ndarray, point_count: int) -> np.ndarray:
    """Per tetrahedron: its wrapped point ids in order plus each corner's tile offset
    relative to the first corner, equal for every translated copy."""
    order = np.argsort(simplices % point_count, axis=1)
    ordered = np.take_along_axis(simplices, order, axis=1)
    ids = ordered % point_count
    tiles = _OFFSETS[ordered // point_count].astype(np.int64)
    relative = (tiles - tiles[:, :1, :]).reshape(len(simplices), -1)
    return np.column_stack([ids, relative])
