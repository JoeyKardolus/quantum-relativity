"""Joint cells: the woven cells cut from the skeleton's tetrahedra, every joint a cell with its own corners (rule 2).

Contract: `joint_cells(skeleton_tets)` returns a CellComplex (cell_complex.py) for
positively oriented skeleton tetrahedra (n, 4). The skeleton is the tetrahedral mesh
first_mesh.py cut once; it decides only which cells neighbour which. Inside each
skeleton tetrahedron t, with barycentric coordinates (l0, l1, l2, l3) and a small
share d, the space is cut by the planes l = d into:
  energy volume  all l >= d: a tetrahedron with its own four corners, the copies of
                 t's corners pulled toward its centre (keys kind 0)
  joint half     l_s < d alone: the slab between t's face s and the energy volume. The
                 halves on the two sides of an inner face make that face's joint cell,
                 a wedge whose two big sides are exactly faces of its two energy volumes;
                 its middle (on the skeleton face, keys kind 1) is inside it. On an outer
                 face the half is left out: the energy volume's outer face borders nothing
  seam part      two l < d: the part along a skeleton edge. The parts of all the
                 tetrahedra around an edge make that edge's seam cell, joint material
                 where the joints around the edge meet: every joint cell touches the
                 neighbouring joint cells around the edge only through it, by its thin
                 sides, so the seam cell is the seam the leak flows along
  corner part    three l < d: the part at a skeleton corner. The parts around a corner
                 make its corner cell, joint material where the seams meet
So space is filled with no gaps and no overlaps: an energy volume touches only joint
cells (by its faces) and the outside; a joint cell touches its two energy volumes by its
big sides and seam cells by its thin sides, and other joint cells only along edges.
Each interface is a triangle (energy | joint) or a four-sided piece, split into two
triangles by the diagonal through its lowest-numbered corner; every polygon is oriented
outward from its first cell once, in a reference tetrahedron, and positively oriented
tetrahedra keep that orientation. A skeleton corner itself is a corner only where it is
on the outside (inside it lies within its corner cell). Positions come from
place_corners.py; this is topology only.
"""
from __future__ import annotations

import numpy as np

from physics.cells.cell_complex import KIND_CORNER, KIND_JOINT, KIND_SEAM, KIND_VOLUME, CellComplex
from physics.cells.face_joints import face_joints
from physics.cells.tet_faces import tet_faces

_SHARE = 0.1
_REFERENCE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def joint_cells(skeleton_tets: np.ndarray) -> CellComplex:
    """The cell complex of energy volumes, joint cells, seam cells and corner cells over the skeleton."""
    tets = np.asarray(skeleton_tets, dtype=np.int64)
    count = len(tets)
    skeleton_faces, skeleton_face_cells = tet_faces(tets)
    joints, skeleton_face_joint = face_joints(skeleton_face_cells)
    width = int(tets.max()) + 1 if count else 1
    slot_joint = _slot_joints(tets, skeleton_faces, skeleton_face_joint, width)
    edges, slot_edge = _edges(tets, width)
    first_seam = count + len(joints)
    first_corner = first_seam + len(edges)
    rings, sides = [], []
    for polygon in _template():
        ring_keys = np.stack([_keys(tets, label) for label in polygon["labels"]], axis=1)
        first = _piece_cells(polygon["first"], slot_joint, slot_edge, tets, first_seam, first_corner)
        second = _second_cells(polygon, slot_joint, slot_edge, tets, first_seam, first_corner)
        kept = second > -2
        flipped = kept & (first < 0)
        first, second = np.where(flipped, second, first), np.where(flipped, -1, second)
        ring_keys[flipped] = ring_keys[flipped][:, ::-1]
        rings.append(ring_keys[kept])
        sides.append(np.column_stack([first[kept], second[kept]]))
    own = np.stack([_keys(tets, ("T", slot)) for slot in range(4)], axis=1)
    every = np.concatenate([own.reshape(-1, 5)] + [ring.reshape(-1, 5) for ring in rings])
    keys, number = np.unique(every, axis=0, return_inverse=True)
    number = number.ravel()
    own_tets = number[:own.size // 5].reshape(count, 4)
    faces, face_cells = _triangles(rings, sides, number[own.size // 5:])
    cell_kind = np.concatenate([np.full(count, KIND_VOLUME), np.full(len(joints), KIND_JOINT),
                                np.full(len(edges), KIND_SEAM), np.full(width, KIND_CORNER)])
    return CellComplex(keys=keys, tets=own_tets, faces=faces, face_cells=face_cells,
                       face_joint=_face_joint(face_cells, count, len(joints)), cell_kind=cell_kind,
                       cell_skeleton=_cell_skeleton(tets, joints, skeleton_faces, skeleton_face_joint,
                                                    edges, width),
                       joints=joints, slot_joint=slot_joint, skeleton_tets=tets, skeleton_faces=skeleton_faces,
                       skeleton_face_cells=skeleton_face_cells, skeleton_face_joint=skeleton_face_joint)


def _slot_joints(tets: np.ndarray, faces: np.ndarray, face_joint: np.ndarray, width: int) -> np.ndarray:
    """Per tetrahedron and slot s, the joint on the face opposite s (-1 outside)."""
    face_keys = _code(np.sort(faces, axis=1), width)
    order = np.argsort(face_keys)
    slot_joint = np.empty((len(tets), 4), dtype=np.int64)
    for slot in range(4):
        opposite = np.sort(np.delete(tets, slot, axis=1), axis=1)
        row = order[np.searchsorted(face_keys[order], _code(opposite, width))]
        slot_joint[:, slot] = face_joint[row]
    return slot_joint


def _edges(tets: np.ndarray, width: int) -> tuple[np.ndarray, dict]:
    """Every skeleton edge once (sorted pairs), and per slot pair the edge row of each tetrahedron."""
    pairs = [(one, two) for one in range(4) for two in range(one + 1, 4)]
    ends = {pair: np.sort(tets[:, list(pair)], axis=1) for pair in pairs}
    codes = np.unique(np.concatenate([_code(end, width) for end in ends.values()]))
    slot_edge = {pair: np.searchsorted(codes, _code(end, width)) for pair, end in ends.items()}
    edges = np.column_stack([codes // width, codes % width]).astype(np.int64).reshape(-1, 2)
    return edges, slot_edge


def _code(rows: np.ndarray, width: int) -> np.ndarray:
    """One int64 per row of small non-negative ints."""
    code = np.zeros(len(rows), dtype=np.int64)
    for column in range(rows.shape[1]):
        code = code * width + rows[:, column]
    return code


def _keys(tets: np.ndarray, label: tuple) -> np.ndarray:
    """Per tetrahedron the (5,) key of one template point."""
    count = len(tets)
    key = np.full((count, 5), -1, dtype=np.int64)
    kind, near = label[0], label[1]
    key[:, 1] = tets[:, near]
    if kind == "T":
        key[:, 0] = 0
        key[:, 2:5] = np.sort(np.delete(tets, near, axis=1), axis=1)
    elif kind == "F":
        key[:, 0] = 1
        key[:, 2:4] = np.sort(tets[:, [slot for slot in range(4) if slot not in (near, label[2])]], axis=1)
    elif kind == "E":
        key[:, 0] = 2
        key[:, 2] = tets[:, label[2]]
    else:
        key[:, 0] = 3
    return key


def _piece_cells(piece: tuple, slot_joint: np.ndarray, slot_edge: dict, tets: np.ndarray,
                 first_seam: int, first_corner: int) -> np.ndarray:
    """Per tetrahedron the cell a template piece belongs to (-1 for a left-out outer joint half)."""
    count = len(tets)
    if piece[0] == "V":
        return np.arange(count, dtype=np.int64)
    if piece[0] == "P":
        joint = slot_joint[:, piece[1]]
        return np.where(joint >= 0, count + joint, -1)
    if piece[0] == "D":
        return first_seam + slot_edge[(piece[1], piece[2])]
    return first_corner + tets[:, piece[1]]


def _second_cells(polygon: dict, slot_joint: np.ndarray, slot_edge: dict, tets: np.ndarray,
                  first_seam: int, first_corner: int) -> np.ndarray:
    """Per tetrahedron the polygon's second cell: a piece, -1 the outside, -2 not a face (inside one cell)."""
    if polygon["second"] is not None:
        return _piece_cells(polygon["second"], slot_joint, slot_edge, tets, first_seam, first_corner)
    outer = slot_joint[:, polygon["on_face"]] < 0
    return np.where(outer, -1, -2)


def _triangles(rings: list, sides: list, number: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of the polygons (four-sided ones cut through their lowest corner) and their cells."""
    faces, cells, start = [], [], 0
    for ring, side in zip(rings, sides):
        size = ring.shape[1]
        corners = number[start:start + ring.size // 5].reshape(len(ring), size)
        start += ring.size // 5
        if size == 3:
            faces.append(corners)
            cells.append(side)
            continue
        through_first = np.minimum(corners[:, 0], corners[:, 2]) < np.minimum(corners[:, 1], corners[:, 3])
        turned = np.where(through_first[:, None], corners, np.roll(corners, -1, axis=1))
        faces += [turned[:, [0, 1, 2]], turned[:, [0, 2, 3]]]
        cells += [side, side]
    return (np.concatenate(faces).astype(np.int64).reshape(-1, 3),
            np.concatenate(cells).astype(np.int64).reshape(-1, 2))


def _face_joint(face_cells: np.ndarray, count: int, joint_count: int) -> np.ndarray:
    """Per triangle the joint whose joint cell is one of its sides, else -1."""
    first, second = face_cells[:, 0], face_cells[:, 1]
    first_is = (first >= count) & (first < count + joint_count)
    second_is = (second >= count) & (second < count + joint_count)
    return np.where(first_is, first - count, np.where(second_is, second - count, -1)).astype(np.int64)


def _cell_skeleton(tets: np.ndarray, joints: np.ndarray, faces: np.ndarray, face_joint: np.ndarray,
                   edges: np.ndarray, width: int) -> np.ndarray:
    """Per cell the skeleton corners of its place, -1 padded."""
    joint_face = np.full((len(joints), 4), -1, dtype=np.int64)
    inside = face_joint >= 0
    joint_face[face_joint[inside], :3] = faces[inside]
    edge_rows = np.full((len(edges), 4), -1, dtype=np.int64)
    edge_rows[:, :2] = edges
    corner_rows = np.full((width, 4), -1, dtype=np.int64)
    corner_rows[:, 0] = np.arange(width)
    return np.concatenate([tets, joint_face, edge_rows, corner_rows])


def _template() -> list[dict]:
    """The polygons cut in one tetrahedron, as template point labels, oriented outward from their first piece."""
    polygons = []
    for opposite in range(4):
        face = [slot for slot in range(4) if slot != opposite]
        polygons.append({"labels": [("T", slot) for slot in face], "first": ("V",),
                         "second": ("P", opposite)})
        for other in face:
            one, two = sorted(slot for slot in face if slot != other)
            polygons.append({"labels": [("T", one), ("T", two), ("F", two, opposite), ("F", one, opposite)],
                             "first": ("D", one, two), "second": ("P", opposite)})
    for one in range(4):
        for two in range(one + 1, 4):
            rest = [slot for slot in range(4) if slot not in (one, two)]
            for end, far in ((one, two), (two, one)):
                polygons.append({"labels": [("T", end), ("F", end, rest[0]), ("E", end, far),
                                            ("F", end, rest[1])],
                                 "first": ("D", one, two), "second": ("C", end)})
            for zero in rest:
                polygons.append({"labels": [("E", one, two), ("E", two, one), ("F", two, zero),
                                            ("F", one, zero)],
                                 "first": ("D", one, two), "second": None, "on_face": zero})
    for corner in range(4):
        for zero in range(4):
            if zero == corner:
                continue
            side_one, side_two = (slot for slot in range(4) if slot not in (corner, zero))
            polygons.append({"labels": [("O", corner), ("E", corner, side_one), ("F", corner, zero),
                                        ("E", corner, side_two)],
                             "first": ("C", corner), "second": None, "on_face": zero})
    return [_outward(polygon) for polygon in polygons]


def _outward(polygon: dict) -> dict:
    """The polygon with its labels turned, if needed, so it faces out of its first piece (reference tetrahedron)."""
    points = np.array([_reference(label) for label in polygon["labels"]])
    normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    centre = np.mean([_reference(label) for label in _piece_labels(polygon["first"])], axis=0)
    if float(normal @ (points.mean(axis=0) - centre)) < 0:
        polygon = dict(polygon, labels=polygon["labels"][::-1])
    return polygon


def _reference(label: tuple) -> np.ndarray:
    """A template point's position in the reference tetrahedron at share _SHARE."""
    weights = np.zeros(4)
    kind, near = label[0], label[1]
    if kind == "T":
        weights[:] = _SHARE
    elif kind == "F":
        weights[:] = _SHARE
        weights[label[2]] = 0.0
    elif kind == "E":
        weights[label[2]] = _SHARE
    weights[near] = 0.0
    weights[near] = 1.0 - weights.sum()
    return weights @ _REFERENCE


def _piece_labels(piece: tuple) -> list[tuple]:
    """The template points of one piece."""
    if piece[0] == "V":
        return [("T", slot) for slot in range(4)]
    if piece[0] == "P":
        face = [slot for slot in range(4) if slot != piece[1]]
        return [("T", slot) for slot in face] + [("F", slot, piece[1]) for slot in face]
    if piece[0] == "D":
        one, two = piece[1], piece[2]
        rest = [slot for slot in range(4) if slot not in (one, two)]
        return ([("T", one), ("T", two), ("E", one, two), ("E", two, one)]
                + [("F", end, zero) for end in (one, two) for zero in rest])
    corner = piece[1]
    others = [slot for slot in range(4) if slot != corner]
    return ([("T", corner), ("O", corner)] + [("F", corner, zero) for zero in others]
            + [("E", corner, other) for other in others])
