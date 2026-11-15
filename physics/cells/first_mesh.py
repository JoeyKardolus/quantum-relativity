"""First mesh: the starting tetrahedra, cut once (Delaunay) from a jittered lattice in a cube.

Contract: `first_mesh(cells, rng)` returns (corners, tets) for about `cells` energy
volumes in the open cube [0, side]^3, side = cells ** (1/3) (so the mean tetrahedron
holds volume 1, the background quantum's volume, rest_volume.py):
  points   a body-centred cubic lattice of spacing SPACING (12 ** (1/3): its Delaunay
           cells are 12 tetrahedra of volume 1 per lattice cube), one layer past the
           cube on every side, each point moved by JITTER (0.15) times the spacing
           uniformly at random per axis (seeded by rng), so the grid is irregular
           and picks out no direction exactly
  cut      scipy.spatial.Delaunay of those points, once; a tetrahedron is kept when
           its centroid lies in the cube (the walls are therefore a little ragged)
  slivers  a kept tetrahedron with an outer face (one no other kept tetrahedron
           shares) and quality below SLIVER (0.2) is dropped, repeatedly, until no
           such one is left; quality is 6 sqrt(2) V / (rms edge)^3, 1 for a regular
           tetrahedron. A sliver inside is kept (dropping it would leave a hole);
           at the default jitter there are about 6 per 1000 below 0.2
  corners  (k, 3) every point some kept tetrahedron uses, renumbered in order
  tets     (n, 4) int corner indices, each positively oriented
           (det(b - a, c - a, d - a) > 0)
This is the only tessellation the engine ever runs; from here the engine only
moves corners, and changes topology only by joint collapse and volume split.
Raises ValueError for cells < 30.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

SPACING = 12.0 ** (1 / 3)
JITTER = 0.15
SLIVER = 0.2
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def first_mesh(cells: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Corners and positively oriented tetrahedra of a jittered lattice cut once in a cube."""
    if cells < 30:
        raise ValueError("ask for at least 30 cells so the cube has an inside")
    side = float(np.cbrt(cells))
    points = _lattice(side, rng)
    tets = Delaunay(points).simplices.astype(np.int64)
    tets = _oriented(points, tets)
    centre = points[tets].mean(axis=1)
    tets = tets[((centre >= 0) & (centre <= side)).all(axis=1)]
    tets = _peeled(points, tets)
    used = np.unique(tets)
    renumber = np.full(len(points), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    return points[used], renumber[tets]


def _lattice(side: float, rng: np.random.Generator) -> np.ndarray:
    """Body-centred cubic points one layer past the cube, jittered."""
    steps = np.arange(-1, int(np.ceil(side / SPACING)) + 2) * SPACING
    cube = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    points = np.vstack([cube, cube + SPACING / 2])
    return points + rng.uniform(-JITTER, JITTER, points.shape) * SPACING


def _oriented(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """The tetrahedra with any negatively oriented one's last two corners swapped."""
    corner = points[tets]
    signed = np.einsum("ij,ij->i", corner[:, 1] - corner[:, 0],
                       np.cross(corner[:, 2] - corner[:, 0], corner[:, 3] - corner[:, 0]))
    flipped = tets.copy()
    flipped[signed < 0, 2], flipped[signed < 0, 3] = tets[signed < 0, 3], tets[signed < 0, 2]
    return flipped


def _quality(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Per tetrahedron 6 sqrt(2) V / (rms edge)^3: 1 regular, near 0 a sliver."""
    corner = points[tets]
    volume = np.abs(np.einsum("ij,ij->i", corner[:, 1] - corner[:, 0],
                              np.cross(corner[:, 2] - corner[:, 0], corner[:, 3] - corner[:, 0]))) / 6
    squared = np.mean([((corner[:, one] - corner[:, two]) ** 2).sum(axis=1) for one, two in _EDGES],
                      axis=0)
    return 6 * np.sqrt(2) * volume / squared ** 1.5


def _peeled(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """The tetrahedra left after slivers with an outer face are dropped, repeatedly."""
    poor = _quality(points, tets) < SLIVER
    while True:
        keys = np.sort(tets[:, _FACES].reshape(-1, 3), axis=1)
        _, slot, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        outer = (count[slot.ravel()] == 1).reshape(-1, 4).any(axis=1)
        dropped = poor & outer
        if not dropped.any():
            return tets
        tets, poor = tets[~dropped], poor[~dropped]
