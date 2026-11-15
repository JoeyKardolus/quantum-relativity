"""Place corners: where each corner of the woven cells sits, built from the skeleton and the joints' thicknesses.

Contract: `place_corners(keys, skeleton_corners, skeleton_tets, offsets)` returns (k, 3)
positions for corner keys (cell_complex.py). offsets (n, 4) is per skeleton tetrahedron
and slot s how far its energy volume's face opposite s stands back from the skeleton
face (joint_offsets.py: half its joint's thickness); per skeleton edge the width e is,
over the tetrahedra around it, the smallest mean offset of their two faces on it: a seam
is as narrow as its most squeezed joints, and never wider than its smallest tetrahedron
holds.
  kind 0  energy volume t's own copy of corner i: the point at distance offset_s from
          each of t's three faces through i (the planes of the energy volume's faces),
          in barycentric terms l_s = offset_s / height_s
  kind 1  on skeleton face (i, j, k) near i: in the face, at distance e_ij from the line
          ij and e_ik from the line ik (shared by the face's two tetrahedra)
  kind 2  on skeleton edge (i, j) near i: e_ij along it from i
  kind 3  skeleton corner i
So a joint cell starts with its two halves offset_s thick and a seam cell about e wide.
Used when the cells are first cut, for the corners a collapse or split makes new, and
each step for the size the seam and corner cells aim at (cell_targets.py).
Raises ValueError for a key whose tetrahedron or face is not in the skeleton.
"""
from __future__ import annotations

import numpy as np

_PAIRS = [(one, two) for one in range(4) for two in range(one + 1, 4)]


def place_corners(keys: np.ndarray, skeleton_corners: np.ndarray, skeleton_tets: np.ndarray,
                  offsets: np.ndarray) -> np.ndarray:
    """Positions of the keyed corners in the skeleton with these face offsets."""
    points = np.asarray(skeleton_corners, dtype=float)
    width = len(points)
    widths = _edge_widths(skeleton_tets, offsets, width)
    placed = np.zeros((len(keys), 3))
    kind = keys[:, 0]
    copies = kind == 0
    placed[copies] = _copies(keys[copies], points, skeleton_tets, offsets, width)
    on_face = kind == 1
    placed[on_face] = _face_points(keys[on_face], points, widths, width)
    on_edge = kind == 2
    near, far = keys[on_edge, 1], keys[on_edge, 2]
    length = np.linalg.norm(points[far] - points[near], axis=1)
    share = _width_of(widths, near, far, width) / length
    placed[on_edge] = points[near] + share[:, None] * (points[far] - points[near])
    corner = kind == 3
    placed[corner] = points[keys[corner, 1]]
    return placed


def _edge_widths(tets: np.ndarray, offsets: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Per skeleton edge (codes sorted) its width: the smallest mean offset of the two faces on it, around it."""
    codes, values = [], []
    for one, two in _PAIRS:
        rest = [slot for slot in range(4) if slot not in (one, two)]
        ends = np.sort(tets[:, [one, two]], axis=1)
        codes.append(ends[:, 0] * width + ends[:, 1])
        values.append(offsets[:, rest].mean(axis=1))
    codes, values = np.concatenate(codes), np.concatenate(values)
    unique, slot = np.unique(codes, return_inverse=True)
    smallest = np.full(len(unique), np.inf)
    np.minimum.at(smallest, slot.ravel(), values)
    return unique, smallest


def _width_of(widths: tuple, near: np.ndarray, far: np.ndarray, width: int) -> np.ndarray:
    """The width of each edge (near, far)."""
    codes, smallest = widths
    wanted = np.minimum(near, far) * width + np.maximum(near, far)
    row = np.searchsorted(codes, wanted)
    if len(wanted) and ((row >= len(codes)) | (codes[np.minimum(row, len(codes) - 1)] != wanted)).any():
        raise ValueError("place_corners: an edge key is not an edge of the skeleton")
    return smallest[row]


def _copies(keys: np.ndarray, points: np.ndarray, tets: np.ndarray, offsets: np.ndarray,
            width: int) -> np.ndarray:
    """Each energy volume's own copy of a corner, at its faces' offsets."""
    tet_codes = _sorted_code(np.sort(tets, axis=1), width)
    order = np.argsort(tet_codes)
    wanted = _sorted_code(np.sort(keys[:, 1:5], axis=1), width)
    found = np.searchsorted(tet_codes[order], wanted)
    if len(wanted) and ((found >= len(order)) | (tet_codes[order][np.minimum(found, len(order) - 1)]
                                                 != wanted)).any():
        raise ValueError("place_corners: a copy's tetrahedron is not in the skeleton")
    row = order[found]
    ring = tets[row]
    slot = np.argmax(ring == keys[:, 1:2], axis=1)
    corner = points[ring]
    gradient = _barycentric_gradients(corner)
    weights = offsets[row] * np.linalg.norm(gradient, axis=2)
    weights[np.arange(len(row)), slot] = 0.0
    weights[np.arange(len(row)), slot] = 1.0 - weights.sum(axis=1)
    return np.einsum("ij,ijk->ik", weights, corner)


def _sorted_code(rows: np.ndarray, width: int) -> np.ndarray:
    """One int64 per sorted row of four corners."""
    code = np.zeros(len(rows), dtype=np.int64)
    for column in range(rows.shape[1]):
        code = code * width + rows[:, column]
    return code


def _barycentric_gradients(corner: np.ndarray) -> np.ndarray:
    """Per tetrahedron (m, 4, 3) the gradient of each barycentric coordinate."""
    edges = corner[:, 1:] - corner[:, :1]
    inverse = np.linalg.inv(edges)
    later = np.transpose(inverse, (0, 2, 1))
    return np.concatenate([-later.sum(axis=1, keepdims=True), later], axis=1)


def _face_points(keys: np.ndarray, points: np.ndarray, widths: tuple, width: int) -> np.ndarray:
    """Each point on a skeleton face near a corner, at the two edges' widths in from them."""
    near, one, two = keys[:, 1], keys[:, 2], keys[:, 3]
    origin = points[near]
    towards_one, towards_two = points[one] - origin, points[two] - origin
    double_area = np.linalg.norm(np.cross(towards_one, towards_two), axis=1)
    from_line_one = _width_of(widths, near, one, width)
    from_line_two = _width_of(widths, near, two, width)
    share_two = from_line_one * np.linalg.norm(towards_one, axis=1) / double_area
    share_one = from_line_two * np.linalg.norm(towards_two, axis=1) / double_area
    return origin + share_one[:, None] * towards_one + share_two[:, None] * towards_two
