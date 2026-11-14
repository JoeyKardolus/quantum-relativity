"""The shared faces of the 3D power diagram (Laguerre cells) of weighted points. Display only.

Contract: `power_faces(points, weights, box, wanted)` takes (n, 3) points,
n weights (a point's power at q is |q - point|^2 - weight), box (x_min,
y_min, z_min, x_max, y_max, z_max) and wanted, a (k, 2) array of point
pairs whose faces are wanted as polygons. Returns (pairs, polygons): pairs
is an (f, 2) int array, i < j per row, one row per pair of cells sharing a
face of area above zero whose bounding box meets the box (every face,
wanted or not, so a caller can count them); polygons[f] is that face as an (m, 3)
array of corners in order around it, clipped to the box, when the pair is
wanted, else None (also None when clipping leaves less than a triangle).
Every face is square to the line from point i to point j. Method: lift each
point to (x, y, z, x^2 + y^2 + z^2 - weight), take the lower convex hull
(scipy ConvexHull, 4D); each lower facet is a corner of the diagram (a
tetrahedron of the regular triangulation), and the corners of the
tetrahedra around one triangulation edge (i, j), sorted by angle around
that edge, are the face between cells i and j. Eight far ghost points keep
every real face closed and are never returned. A point whose cell is empty
has no face.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import ConvexHull

GHOST_REACH = 100.0
TINY = 1e-9
EDGE_CORNERS = np.array(list(itertools.combinations(range(4), 2)))


def power_faces(points: np.ndarray, weights: np.ndarray,
                box: tuple[float, float, float, float, float, float],
                wanted: np.ndarray) -> tuple[np.ndarray, list]:
    """Every face two power cells share in the box, and the polygons of the wanted ones."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    count = len(points)
    if count < 2:
        return np.zeros((0, 2), dtype=np.int64), []
    lows, highs = np.asarray(box[:3], dtype=float), np.asarray(box[3:], dtype=float)
    tetrahedra, corners = _lower_hull(points, np.asarray(weights, dtype=float), lows, highs)
    pairs, ring_corners, starts, outside = _rings(count, points, tetrahedra, corners, lows, highs)
    wanted = np.sort(np.asarray(wanted, dtype=np.int64).reshape(-1, 2), axis=1)
    asked = np.isin(pairs[:, 0] * count + pairs[:, 1], wanted[:, 0] * count + wanted[:, 1])
    polygons = []
    for index in range(len(pairs)):
        ring = None
        if asked[index]:
            ring = ring_corners[starts[index]:starts[index + 1]]
            if outside[index]:
                ring = _clipped(ring, lows, highs)
        polygons.append(ring)
    return pairs, polygons


def _lower_hull(points, weights, lows, highs):
    centre = (lows + highs) / 2
    reach = GHOST_REACH * max(float((highs - lows).max()), 1.0)
    ghosts = centre + reach * np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    every = np.vstack([points, ghosts])
    every_weight = np.concatenate([weights, np.zeros(len(ghosts))])
    lifted = np.column_stack([every, (every ** 2).sum(axis=1) - every_weight])
    hull = ConvexHull(lifted, qhull_options="Qt")
    normal = hull.equations
    lower = normal[:, 3] < -TINY
    corners = -normal[lower, :3] / (2.0 * normal[lower, 3:4])
    return hull.simplices[lower], corners


def _rings(count, points, tetrahedra, corners, lows, highs):
    """Faces of area above zero whose bounds meet the box: pairs, ring corners, starts, outside.

    The corners of the tetrahedra around each triangulation edge between two real
    points are sorted by angle around that edge; corners closer than TINY to the
    next one in their ring are dropped.
    """
    ends = np.sort(tetrahedra[:, EDGE_CORNERS], axis=2).reshape(-1, 2)
    corner = np.repeat(corners, len(EDGE_CORNERS), axis=0)
    real = ends[:, 1] < count
    ends, corner = ends[real], corner[real]
    keys = ends[:, 0].astype(np.int64) * count + ends[:, 1]
    unique, group = np.unique(keys, return_inverse=True)
    group = group.ravel()
    centre = _group_mean(group, corner, len(unique))
    axis = points[unique % count] - points[unique // count]
    axis /= np.maximum(np.linalg.norm(axis, axis=1), TINY)[:, None]
    seed = np.where(np.abs(axis[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    across = np.cross(axis, seed)
    across /= np.maximum(np.linalg.norm(across, axis=1), TINY)[:, None]
    other = np.cross(axis, across)
    offset = corner - centre[group]
    angle = np.arctan2((offset * other[group]).sum(axis=1), (offset * across[group]).sum(axis=1))
    order = np.lexsort((angle, group))
    corner, group = corner[order], group[order]
    following = _following(group)
    distinct = np.linalg.norm(corner[following] - corner, axis=1) > TINY
    corner, group = corner[distinct], group[distinct]
    following = _following(group)
    spokes = corner - centre[group]
    twice_area = np.stack([np.bincount(group, column, len(unique)) for column in
                           np.cross(spokes, spokes[following]).T], axis=1)
    area = 0.5 * np.linalg.norm(twice_area, axis=1)
    sizes = np.bincount(group, minlength=len(unique))
    away = ((corner < lows) | (corner > highs)).any(axis=1)
    outside = np.bincount(group, away, len(unique)) > 0
    smallest = np.full((len(unique), 3), np.inf)
    largest = np.full((len(unique), 3), -np.inf)
    np.minimum.at(smallest, group, corner)
    np.maximum.at(largest, group, corner)
    reaching = ((smallest <= highs) & (largest >= lows)).all(axis=1)
    kept = (sizes >= 3) & (area > TINY) & reaching
    pairs = np.stack([unique // count, unique % count], axis=1)
    kept_starts = np.concatenate([[0], np.cumsum(sizes[kept])])
    return pairs[kept], corner[kept[group]], kept_starts, outside[kept]


def _group_mean(group, values, size):
    totals = np.stack([np.bincount(group, values[:, column], size) for column in range(3)], axis=1)
    return totals / np.maximum(np.bincount(group, minlength=size), 1)[:, None]


def _following(group):
    """Index of the next corner in the same ring, the last wrapping to the first."""
    index = np.arange(len(group))
    following = index + 1
    last = np.flatnonzero(np.diff(group, append=-1) != 0)
    first = np.concatenate([[0], last[:-1] + 1]) if len(last) else last
    following[last] = first
    return following


def _deduplicated(ring):
    steps = np.linalg.norm(ring - np.roll(ring, -1, axis=0), axis=1)
    return ring[steps > TINY]


def _clipped(ring, lows, highs):
    """Sutherland-Hodgman against the six box sides; None when less than a triangle is left."""
    for axis in range(3):
        for limit, keep_above in ((lows[axis], True), (highs[axis], False)):
            ring = _clip_side(ring, axis, limit, keep_above)
            if len(ring) < 3:
                return None
    return _deduplicated(ring) if len(_deduplicated(ring)) >= 3 else None


def _clip_side(ring, axis, limit, keep_above):
    inside = ring[:, axis] >= limit if keep_above else ring[:, axis] <= limit
    kept = []
    for index in range(len(ring)):
        here, after = ring[index], ring[(index + 1) % len(ring)]
        here_in, after_in = inside[index], inside[(index + 1) % len(ring)]
        if here_in:
            kept.append(here)
        if here_in != after_in:
            share = (limit - here[axis]) / (after[axis] - here[axis])
            kept.append(here + share * (after - here))
    return np.array(kept).reshape(-1, 3)


def _area(ring):
    if len(ring) < 3:
        return 0.0
    spokes = ring - ring.mean(axis=0)
    return 0.5 * float(np.linalg.norm(np.cross(spokes, np.roll(spokes, -1, axis=0)).sum(axis=0)))
