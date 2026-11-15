"""Split cell: cut one cell in two by a new face through its middle (volume split, bible rule 8).

Contract: `split_cell(corners, faces, face_cells, face_origin, cell, new_cell)` returns
(corners, faces, face_cells, face_origin, share) or None when the cut is not clean.
  plane    through the mean of the cell's corners, across its longest direction (the
           corners' principal axis, largest covariance eigenvalue); corners on the
           plane's positive side (distance >= 0) go to new_cell, the rest stay
  faces    a face of the cell wholly on one side goes to that side's cell; a face the
           plane crosses is cut in two at new corners on its crossing edges (one new
           corner per crossing edge, shared by both faces on that edge), each piece
           keeping the face's ring order, its other cell and its old joint
           (face_origin), so a neighbour's face cut in two keeps its joint's thickness
  tear     a new face, ring of the new corners in order, first cell `cell`, second
           new_cell, face_origin -1: the new joint between the two halves
  share    the volume of the part kept by `cell` over the two halves' volumes, in
           (0, 1), so the caller shares energy by volume
Not clean, and None: a face crossed more than twice, a cross-section of more than one
loop (a cell bent around the plane), nothing on one side, or a half whose volume is
not positive (a warped cell). The mesh is never left
half cut. Corners are appended; none is moved. On flat faces every other cell keeps its volume
exactly and the halves sum to the old cell; a warped face cut in two is triangulated
afresh, so its neighbour's measured volume can shift a little (the corners then relax).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def split_cell(corners: np.ndarray, faces: Sequence[np.ndarray], face_cells: np.ndarray,
               face_origin: np.ndarray, cell: int, new_cell: int):
    """The mesh with `cell` cut across its longest direction, or None when the cut is not clean."""
    mine = np.flatnonzero((face_cells == cell).any(axis=1))
    outward = [faces[index] if face_cells[index, 0] == cell else faces[index][::-1]
               for index in mine]
    used = np.unique(np.concatenate(outward))
    centre = corners[used].mean(axis=0)
    normal = np.linalg.eigh(np.cov((corners[used] - centre).T))[1][:, -1]
    distance = np.zeros(len(corners))
    distance[used] = (corners[used] - centre) @ normal
    positive = distance >= 0
    if positive[used].all() or not positive[used].any():
        return None
    added: dict[tuple[int, int], int] = {}
    new_positions: list[np.ndarray] = []

    def crossing(first: int, second: int) -> int:
        key = (min(first, second), max(first, second))
        if key not in added:
            share = distance[first] / (distance[first] - distance[second])
            new_positions.append(corners[first] + share * (corners[second] - corners[first]))
            added[key] = len(corners) + len(new_positions) - 1
        return added[key]

    next_corner: dict[int, int] = {}
    pieces = []
    for index, ring in zip(mine, outward):
        cut = _cut(ring, positive, crossing)
        if cut is None:
            return None
        if cut[2] is not None:
            entering, leaving = cut[2]
            next_corner[leaving] = entering
        original = faces[index]
        pieces.append(_cut(original, positive, crossing)[:2] if cut[2] is not None
                      else ((original, None) if not positive[original].any() else (None, original)))
    tear = _loop(next_corner)
    if tear is None:
        return None
    new_faces, new_cells, new_origin = list(faces), face_cells.copy(), face_origin.copy()
    for index, (negative_piece, positive_piece) in zip(mine, pieces):
        moved = np.where(face_cells[index] == cell, new_cell, face_cells[index])
        if negative_piece is None:
            new_cells[index] = moved
            continue
        new_faces[index] = negative_piece
        if positive_piece is not None:
            new_faces.append(positive_piece)
            new_cells = np.vstack([new_cells, moved])
            new_origin = np.append(new_origin, face_origin[index])
    new_faces.append(tear)
    new_cells = np.vstack([new_cells, [cell, new_cell]])
    new_origin = np.append(new_origin, -1)
    all_corners = np.vstack([corners, np.array(new_positions).reshape(-1, 3)])
    kept, other = (_enclosed(all_corners, [ring for ring, cells in zip(new_faces, new_cells)
                                            if half in cells],
                             [cells for cells in new_cells if half in cells], half)
                   for half in (cell, new_cell))
    if kept <= 0 or other <= 0:
        return None
    return all_corners, new_faces, new_cells, new_origin, kept / (kept + other)


def _cut(ring: np.ndarray, positive: np.ndarray, crossing):
    """(negative piece, positive piece, (entering, leaving) new corners) of a ring, None past two crossings.

    A ring wholly on one side gives (ring, None, None) or (None, ring, None)."""
    side = positive[ring]
    if side.all():
        return None, ring, None
    if not side.any():
        return ring, None, None
    following = np.roll(ring, -1)
    if int((side != positive[following]).sum()) != 2:
        return None
    negative_piece, positive_piece = [], []
    entering = leaving = -1
    for first, second, first_side, second_side in zip(ring, following, side, positive[following]):
        (positive_piece if first_side else negative_piece).append(int(first))
        if first_side != second_side:
            new = crossing(int(first), int(second))
            negative_piece.append(new)
            positive_piece.append(new)
            if first_side:
                leaving = new
            else:
                entering = new
    return (np.array(negative_piece, dtype=np.int64), np.array(positive_piece, dtype=np.int64),
            (entering, leaving))


def _loop(next_corner: dict[int, int]) -> np.ndarray | None:
    """The tear's ring following leaving -> entering corners, None unless one loop takes them all."""
    if len(next_corner) < 3:
        return None
    start = next(iter(next_corner))
    ring = [start]
    while len(ring) <= len(next_corner):
        following = next_corner.get(ring[-1])
        if following is None:
            return None
        if following == start:
            break
        ring.append(following)
    if len(ring) != len(next_corner):
        return None
    return np.array(ring, dtype=np.int64)


def _enclosed(corners: np.ndarray, rings: list[np.ndarray], cells, cell: int) -> float:
    """Volume `cell` encloses: its rings fanned from their first corner, signed by its side."""
    centre = corners.mean(axis=0)
    total = 0.0
    for ring, pair in zip(rings, cells):
        sign = 1.0 if pair[0] == cell else -1.0
        points = corners[ring] - centre
        total += sign * float(np.einsum("ij,ij->", np.broadcast_to(points[0], points[1:-1].shape),
                                        np.cross(points[1:-1], points[2:]))) / 6
    return total
