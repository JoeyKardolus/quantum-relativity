"""First mesh: the starting cell geometry, cut once from a power tessellation of a cube.

Contract: `first_mesh(points, side, targets)` returns (corners, faces, face_cells) for
n points in the open cube [0, side]^3, not wrapped, cut as the power (Laguerre)
tessellation whose cells come near `targets` (n,), the volume each cell should have
(scaled here to fill the cube): FIT_ROUNDS (30) rounds in which every point moves to
its cell's vertex mean (kept EDGE_GAP * side inside the walls) and the power weights
take one damped Newton step, w_i += DAMPING (0.5) * (target_i - V_i) / sum_faces (area / (2 * gap)), each
change kept within STEP_LIMIT (0.3) times the cell's target radius squared (an empty
cell's weight rises by that limit), and up to MORE_ROUNDS (30) more while a cell
is empty. The faces stay flat; relax_corners() finishes the
fit. Returned:
  corners     (k, 3) every Voronoi vertex once, shared by all cells that meet there
              (vertices voro++ gives per cell closer than MERGE_GAP * side are one)
  faces       list of int arrays: each face's corner ring, outward from its first cell
  face_cells  (f, 2) the face's first cell and its second; a face on the cube's wall
              has second cell -1 (it borders nothing)
A face shared by cells a < b is kept once, with a's ring (outward from a). These are
the only tessellations the engine ever runs, all at the start (voro++ through pyvoro2's native call,
pyvoro2 0.8.0 pinned); from here the engine only moves corners, and changes topology
only by joint collapse and volume split. Raises ValueError for non-finite points,
points outside the cube, or a cell still empty after the fit.
"""
from __future__ import annotations

import numpy as np
from pyvoro2 import _core
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MERGE_GAP = 1e-9
FIT_ROUNDS = 30
EDGE_GAP = 1e-6
MORE_ROUNDS = 30
DAMPING = 0.5
STEP_LIMIT = 0.3
_RETURN_CORNERS_AND_FACES = (True, False, True)
_INITIAL_MEMORY = 8


def first_mesh(points: np.ndarray, side: float,
               targets: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Shared corners, faces and each face's two cells of a power tessellation fitted to `targets`."""
    points = np.ascontiguousarray(points, dtype=float)
    if not np.isfinite(points).all() or (points < 0).any() or (points > side).any():
        raise ValueError("points must be finite and inside the cube [0, side]^3")
    wanted = np.asarray(targets, dtype=float) * side ** 3 / float(np.sum(targets))
    weights = np.zeros(len(points))
    for _ in range(FIT_ROUNDS):
        points, weights = _fitted(points, weights, side, wanted)
    records = _tessellated(points, weights, side)
    for _ in range(MORE_ROUNDS):
        if len(records) == len(points) and all(record["faces"] for record in records):
            break
        points, weights = _fitted(points, weights, side, wanted)
        records = _tessellated(points, weights, side)
    if len(records) != len(points):
        raise ValueError("a cell is still empty after the fit")
    local, offsets, cells = [], [0], []
    for record in records:
        if not record["faces"]:
            raise ValueError(f"voro++ left cell {record['id']} empty")
        local.append(np.asarray(record["vertices"], dtype=float).reshape(-1, 3))
        offsets.append(offsets[-1] + len(local[-1]))
        cells.append(record)
    corners, global_id = _shared_corners(np.concatenate(local), side)
    faces, face_cells = [], []
    for record, offset, vertices in zip(cells, offsets, local):
        cell = record["id"]
        rings = [np.asarray(face["vertices"], dtype=np.int64) for face in record["faces"]]
        flip = _signed_volume(vertices, rings) < 0
        for face, ring in zip(record["faces"], rings):
            other = face["adjacent_cell"]
            if 0 <= other < cell:
                continue
            ordered = ring[::-1] if flip else ring
            faces.append(global_id[ordered + offset])
            face_cells.append((cell, other if other >= 0 else -1))
    return corners, faces, np.asarray(face_cells, dtype=np.int64)


def _tessellated(points: np.ndarray, weights: np.ndarray, side: float) -> list:
    """voro++'s cell records of the power tessellation of points with weights in the cube."""
    count = len(points)
    blocks = (max(1, int(round(count ** (1 / 3) / 2.5))),) * 3
    return _core.compute_box_power(points, np.arange(count, dtype=np.int32),
                                   np.sqrt(weights - weights.min()), ((0.0, side),) * 3, blocks,
                                   (False, False, False), _INITIAL_MEMORY,
                                   _RETURN_CORNERS_AND_FACES)


def _fitted(points: np.ndarray, weights: np.ndarray, side: float,
            wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points moved to their cells' vertex means, weights one damped Newton round toward `wanted`."""
    volume = np.zeros(len(points))
    moved = points.copy()
    owner, other, here, following = [], [], [], []
    for record in _tessellated(points, weights, side):
        cell = record["id"]
        volume[cell] = record["volume"]
        vertices = np.asarray(record["vertices"], dtype=float).reshape(-1, 3)
        moved[cell] = vertices.mean(axis=0)
        for face in record["faces"]:
            if face["adjacent_cell"] < 0:
                continue
            ring = face["vertices"]
            owner.append(cell)
            other.append(face["adjacent_cell"])
            here.append(vertices[ring])
            following.append(vertices[ring[1:] + ring[:1]])
    sizes = np.fromiter(map(len, here), dtype=np.int64, count=len(here))
    newell = np.cross(np.concatenate(here), np.concatenate(following))
    face_of = np.repeat(np.arange(len(here)), sizes)
    summed = np.stack([np.bincount(face_of, newell[:, axis], len(here)) for axis in range(3)], axis=1)
    owner, other = np.asarray(owner), np.asarray(other)
    gaps = np.maximum(np.linalg.norm(points[other] - points[owner], axis=1), 1e-12)
    rate = np.bincount(owner, np.linalg.norm(summed, axis=1) / 2 / (2 * gaps), len(points))
    limit = STEP_LIMIT * (3 * wanted / (4 * np.pi)) ** (2 / 3)
    change = np.divide(DAMPING * (wanted - volume), rate, out=limit.copy(), where=rate > 0)
    edge = EDGE_GAP * side
    return np.clip(moved, edge, side - edge), weights + np.clip(change, -limit, limit)


def _shared_corners(vertices: np.ndarray, side: float) -> tuple[np.ndarray, np.ndarray]:
    """One corner per group of per-cell vertices that coincide, and each vertex's corner."""
    pairs = cKDTree(vertices).query_pairs(MERGE_GAP * side, output_type="ndarray")
    links = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(vertices), len(vertices)))
    groups, label = connected_components(links, directed=False)
    corners = np.zeros((groups, 3))
    np.add.at(corners, label, vertices)
    corners /= np.bincount(label, minlength=groups)[:, None]
    return corners, label


def _signed_volume(vertices: np.ndarray, rings: list[np.ndarray]) -> float:
    """Signed volume a cell's rings enclose in their given order."""
    centred = vertices - vertices.mean(axis=0)
    total = 0.0
    for ring in rings:
        first = centred[ring[0]]
        for second, third in zip(centred[ring[1:-1]], centred[ring[2:]]):
            total += float(np.dot(first, np.cross(second, third))) / 6
    return total
