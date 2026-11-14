"""The 2D power diagram (Laguerre cells) of weighted points, clipped to a box. Display only.

Contract: `power_cells(points, weights, box)` takes (n, 2) points, n weights
(a point's power at q is |q - point|^2 - weight) and box (x_min, y_min,
x_max, y_max). Returns (polygons, walls): polygons[i] is point i's cell as an
(m, 2) counter-clockwise array clipped to the box, or None when the point
owns no part of the box (a hidden point); walls is a list of (i, j, segment)
with segment a (2, 2) array, one per pair of cells sharing a wall of length
above zero inside the box. Together the polygons tile the box with no gap and
no overlap. Method: lift each point to (x, y, x^2 + y^2 - weight), take the
lower convex hull (scipy ConvexHull, triangulated); each lower facet is a
corner of the diagram, each hull edge between two lower facets a wall. Four
far ghost points keep every real cell bounded and are never returned.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

GHOST_REACH = 100.0
TINY = 1e-9


def power_cells(points: np.ndarray, weights: np.ndarray,
                box: tuple[float, float, float, float]) -> tuple[list, list]:
    """Each point's power cell inside the box, and the walls between them."""
    count = len(points)
    if count == 0:
        return [], []
    corners, lower, simplices, neighbours = _lower_hull(points, weights, box)
    polygons = _polygons(count, simplices, corners, box)
    walls = _walls(count, simplices, neighbours, corners, lower, box)
    return polygons, walls


def _lower_hull(points, weights, box):
    x_min, y_min, x_max, y_max = box
    centre = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    reach = GHOST_REACH * max(x_max - x_min, y_max - y_min, 1.0)
    ghosts = centre + reach * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    every = np.vstack([np.asarray(points, dtype=float), ghosts])
    every_weight = np.concatenate([np.asarray(weights, dtype=float), np.zeros(4)])
    lifted = np.column_stack([every, (every ** 2).sum(axis=1) - every_weight])
    hull = ConvexHull(lifted, qhull_options="Qt")
    normal = hull.equations
    lower = normal[:, 2] < -TINY
    corners = np.full((len(normal), 2), np.nan)
    corners[lower] = -normal[lower, :2] / (2.0 * normal[lower, 2:3])
    return corners, lower, hull.simplices, hull.neighbors


def _polygons(count, simplices, corners, box):
    facet_of = np.repeat(np.arange(len(simplices)), 3)
    owner = simplices.ravel()
    keep = (owner < count) & np.isfinite(corners[facet_of, 0])
    facet_of, owner = facet_of[keep], owner[keep]
    order = np.argsort(owner, kind="stable")
    facet_of, owner = facet_of[order], owner[order]
    starts = np.searchsorted(owner, np.arange(count + 1))
    polygons = []
    for index in range(count):
        ring = corners[facet_of[starts[index]:starts[index + 1]]]
        polygons.append(_clipped_ring(ring, box) if len(ring) >= 3 else None)
    return polygons


def _clipped_ring(ring, box):
    centre = ring.mean(axis=0)
    ring = ring[np.argsort(np.arctan2(ring[:, 1] - centre[1], ring[:, 0] - centre[0]))]
    x_min, y_min, x_max, y_max = box
    if (ring.min(axis=0) < (x_min, y_min)).any() or (ring.max(axis=0) > (x_max, y_max)).any():
        for axis, limit, keep_above in ((0, x_min, True), (0, x_max, False),
                                        (1, y_min, True), (1, y_max, False)):
            ring = _clip_side(ring, axis, limit, keep_above)
            if len(ring) < 3:
                return None
    steps = np.linalg.norm(ring - np.roll(ring, -1, axis=0), axis=1)
    ring = ring[steps > TINY]
    if len(ring) < 3 or _area(ring) <= TINY:
        return None
    return ring


def _clip_side(ring, axis, limit, keep_above):
    """Sutherland-Hodgman against one box side."""
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
    return np.array(kept).reshape(-1, 2)


def _area(ring):
    return 0.5 * abs(float(np.dot(ring[:, 0], np.roll(ring[:, 1], -1))
                           - np.dot(ring[:, 1], np.roll(ring[:, 0], -1))))


def _walls(count, simplices, neighbours, corners, lower, box):
    facets = np.repeat(np.flatnonzero(lower), 3)
    opposite = np.tile(np.arange(3), int(lower.sum()))
    others = neighbours[facets, opposite]
    keep = lower[others] & (others > facets)
    facets, opposite, others = facets[keep], opposite[keep], others[keep]
    triangles = simplices[facets]
    first = triangles[np.arange(len(facets)), (opposite + 1) % 3]
    second = triangles[np.arange(len(facets)), (opposite + 2) % 3]
    real = (first < count) & (second < count)
    starts, ends = corners[facets[real]], corners[others[real]]
    lows, highs = np.minimum(first, second)[real], np.maximum(first, second)[real]
    x_min, y_min, x_max, y_max = box
    inside = ((np.minimum(starts, ends) >= (x_min, y_min)).all(axis=1)
              & (np.maximum(starts, ends) <= (x_max, y_max)).all(axis=1)
              & (np.linalg.norm(ends - starts, axis=1) > TINY))
    walls = []
    for index in range(len(starts)):
        if inside[index]:
            segment = np.array([starts[index], ends[index]])
        else:
            segment = _clipped_segment(starts[index], ends[index], box)
        if segment is not None:
            walls.append((int(lows[index]), int(highs[index]), segment))
    return walls


def _clipped_segment(start, end, box):
    """Liang-Barsky: the part of start-end inside the box, None when shorter than TINY."""
    x_min, y_min, x_max, y_max = box
    delta = end - start
    enter, leave = 0.0, 1.0
    for push, room in ((-delta[0], start[0] - x_min), (delta[0], x_max - start[0]),
                       (-delta[1], start[1] - y_min), (delta[1], y_max - start[1])):
        if abs(push) < TINY:
            if room < 0:
                return None
            continue
        share = room / push
        if push < 0:
            enter = max(enter, share)
        else:
            leave = min(leave, share)
    if enter >= leave:
        return None
    segment = np.array([start + enter * delta, start + leave * delta])
    return segment if np.linalg.norm(segment[1] - segment[0]) > TINY else None
