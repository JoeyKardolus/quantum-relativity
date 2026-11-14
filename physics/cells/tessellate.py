"""Tessellate: a periodic power (Laguerre) tessellation of the working positions.

Contract: `tessellate(positions, weights, side)` returns a Tessellation
(tessellation.py) of n cells filling the periodic cube [0, side)^3 with no gaps:
cell i is every point x whose power |x - p_i|^2 - w_i is lowest at i (nearest
periodic image). A larger weight gives a larger cell; adding one constant to
every weight changes nothing. voro++ computes the cells through pyvoro2's
native call (pyvoro2 0.8.0, LGPL-3+, pinned: its public compute() spends about
two thirds of the time re-validating input this module already checks: finite
values, weights turned into radii sqrt(w - min w), positions wrapped into the
half-open box, no two positions closer than MIN_GAP times the mean spacing).
Volumes come from voro++; centroids and face areas are summed here from each
face's fan of triangles around the cell's mean corner. Seams are the edges:
within a cell, an edge (two consecutive corners of a face) lies on exactly two
of its faces, and the cell with those two faces' other cells are the three around
it. A cell can come out empty
(no faces, volume 0) when its weight is far below its neighbours'; it then has
no joints. Raises ValueError for non-finite input or two positions too close.
"""
from __future__ import annotations

from itertools import chain

import numpy as np
from pyvoro2 import _core
from scipy.spatial import cKDTree

from physics.cells.tessellation import Tessellation

MIN_GAP = 1e-4
_RETURN_CORNERS_AND_FACES = (True, False, True)
_INITIAL_MEMORY = 8


def tessellate(positions: np.ndarray, weights: np.ndarray, side: float) -> Tessellation:
    """The exact periodic power tessellation of `positions` with power `weights` in a cube."""
    wrapped = _checked(positions, weights, side)
    count = len(wrapped)
    blocks = (max(1, int(side / (2.5 * (side ** 3 / count) ** (1 / 3)))),) * 3
    records = _core.compute_box_power(wrapped, np.arange(count, dtype=np.int32),
                                      np.sqrt(weights - weights.min()), ((0.0, side),) * 3,
                                      blocks, (True, True, True), _INITIAL_MEMORY,
                                      _RETURN_CORNERS_AND_FACES)
    volume = np.zeros(count)
    centroid = wrapped.copy()
    face_cell, face_other, corner_index, face_size, cell_corners = [], [], [], [], []
    used_per_cell, offsets = [], []
    offset = 0
    for record in records:
        faces = record["faces"]
        if not faces:
            continue
        cell = record["id"]
        volume[cell] = record["volume"]
        rings = [face["vertices"] for face in faces]
        face_cell.extend([cell] * len(faces))
        face_other.extend([face["adjacent_cell"] for face in faces])
        sizes = list(map(len, rings))
        face_size.extend(sizes)
        corner_index.extend(chain.from_iterable(rings))
        used_per_cell.append(sum(sizes))
        offsets.append(offset)
        cell_corners.extend(record["vertices"])
        offset += len(record["vertices"])
    shifted = (np.asarray(corner_index, dtype=np.int64)
               + np.repeat(np.asarray(offsets, dtype=np.int64), used_per_cell))
    corners = np.array(cell_corners, dtype=float).reshape(-1, 3)[shifted]
    face_cell = np.asarray(face_cell, dtype=np.int64)
    face_other = np.asarray(face_other, dtype=np.int64)
    face_size = np.asarray(face_size, dtype=np.int64)
    face_start = np.concatenate([[0], np.cumsum(face_size)[:-1]]).astype(np.int64)
    face_area, cone_volume, cone_moment = _fan_sums(corners, face_start, face_size,
                                                    _apexes(corners, face_cell, face_size, count),
                                                    face_cell)
    filled = np.bincount(face_cell, cone_volume, count)
    moment = np.stack([np.bincount(face_cell, cone_moment[:, axis], count) for axis in range(3)],
                      axis=1)
    has_shape = filled > 0
    centroid[has_shape] = moment[has_shape] / filled[has_shape, None]
    pairs, face_joint = _pairs(face_cell, face_other)
    lower = (face_joint >= 0) & (face_cell < face_other)
    return Tessellation(volume=volume, centroid=centroid, face_cell=face_cell,
                        face_other=face_other, face_area=face_area, corners=corners,
                        face_start=face_start, face_size=face_size, pairs=pairs,
                        face_joint=face_joint,
                        pair_area=np.bincount(face_joint[lower], face_area[lower], len(pairs)),
                        pair_gap=_gaps(wrapped, pairs, side),
                        seams=_seams(shifted, face_start, face_size, face_cell, face_other))


def _checked(positions: np.ndarray, weights: np.ndarray, side: float) -> np.ndarray:
    """Positions wrapped into [0, side), after checking what voro++ cannot take."""
    if not (np.isfinite(positions).all() and np.isfinite(weights).all() and side > 0):
        raise ValueError("positions, weights and side must be finite, side > 0")
    wrapped = np.ascontiguousarray(np.mod(positions, side), dtype=float)
    wrapped[wrapped >= side] = 0.0
    spacing = (side ** 3 / len(wrapped)) ** (1 / 3)
    if cKDTree(wrapped, boxsize=side).query_pairs(MIN_GAP * spacing):
        raise ValueError("two working positions nearly coincide; voro++ cannot separate them")
    return wrapped


def _apexes(corners: np.ndarray, face_cell: np.ndarray, face_size: np.ndarray,
            count: int) -> np.ndarray:
    """Per cell a point inside it: the mean of its faces' corners (convex cells)."""
    owner = np.repeat(face_cell, face_size)
    total = np.bincount(owner, minlength=count).astype(float)
    sums = np.stack([np.bincount(owner, corners[:, axis], count) for axis in range(3)], axis=1)
    return np.divide(sums, total[:, None], out=np.zeros((count, 3)), where=total[:, None] > 0)


def _fan_sums(corners: np.ndarray, face_start: np.ndarray, face_size: np.ndarray,
              apex: np.ndarray, face_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per face: its area, the volume of the cone from its cell's apex to it, and
    that cone's volume times its centroid (for the cell centroid)."""
    faces = len(face_size)
    triangles = face_size - 2
    owner = np.repeat(np.arange(faces), triangles)
    within = np.arange(triangles.sum()) - np.repeat(np.cumsum(triangles) - triangles, triangles)
    first = corners[face_start[owner]]
    second = corners[face_start[owner] + within + 1]
    third = corners[face_start[owner] + within + 2]
    top = apex[face_cell[owner]]
    cross = np.cross(second - first, third - first)
    area = np.bincount(owner, np.linalg.norm(cross, axis=1) / 2, faces)
    cone = np.abs(np.einsum("ij,ij->i", cross, top - first)) / 6
    middle = (first + second + third + top) / 4
    cone_moment = np.stack([np.bincount(owner, cone * middle[:, axis], faces)
                            for axis in range(3)], axis=1)
    return area, np.bincount(owner, cone, faces), cone_moment


def _pairs(face_cell: np.ndarray, face_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique (a < b) pairs of cells sharing a face, and each face's pair row."""
    ends = np.sort(np.column_stack([face_cell, face_other]), axis=1)
    shared = ends[:, 0] != ends[:, 1]
    pairs, slot = np.unique(ends[shared], axis=0, return_inverse=True)
    face_joint = np.full(len(face_cell), -1, dtype=np.int64)
    face_joint[shared] = slot.ravel()
    return pairs.astype(np.int64).reshape(-1, 2), face_joint


def _seams(vertex: np.ndarray, face_start: np.ndarray, face_size: np.ndarray,
           face_cell: np.ndarray, face_other: np.ndarray) -> np.ndarray:
    """Sorted unique triples (a < b < c) of the three distinct cells around each edge.

    `vertex` is each corner's vertex id, unique per cell, so an edge key (two ids)
    is found only within its own cell, on the two faces of that cell it borders.
    """
    face_of = np.repeat(np.arange(len(face_size)), face_size)
    position = np.arange(len(vertex))
    ring_end = face_start[face_of] + face_size[face_of]
    following = np.where(position + 1 == ring_end, face_start[face_of], position + 1)
    ends = np.sort(np.column_stack([vertex, vertex[following]]), axis=1)
    _, inverse, counts = np.unique(ends, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    on_two = counts[inverse] == 2
    order = np.argsort(inverse[on_two], kind="stable")
    bordering = face_of[on_two][order].reshape(-1, 2)
    triples = np.column_stack([face_cell[bordering[:, 0]], face_other[bordering[:, 0]],
                               face_other[bordering[:, 1]]])
    distinct = ((triples[:, 0] != triples[:, 1]) & (triples[:, 0] != triples[:, 2])
                & (triples[:, 1] != triples[:, 2]))
    return np.unique(np.sort(triples[distinct], axis=1), axis=0).astype(np.int64).reshape(-1, 3)


def _gaps(positions: np.ndarray, pairs: np.ndarray, side: float) -> np.ndarray:
    """Nearest-image distance between the two working positions of each pair."""
    offset = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    offset -= side * np.round(offset / side)
    return np.linalg.norm(offset, axis=1)
