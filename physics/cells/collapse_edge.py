"""Collapse edge: a joint's face gives way along one edge, drawn together into one corner (bible rule 8).

Contract: `collapse_edge(corners, tets, values, edge)` returns None when the collapse
is refused, else (corners, tets, values, kept, gone). values (n, v) are per tetrahedron the
quantities that are handed on (energy, quanta, charge). edge (a, b) is two corners.
  point     b moves into a: to the edge's midpoint when both are inside, onto the one
            on the outside when one is (so the mesh's outer surface does not move);
            refused when both are on the outside
  vanish    every tetrahedron with both corners falls in (it has no volume left);
            its two faces (a, c, d) and (b, c, d) become one, so the tetrahedra
            across them become neighbours. Its values go half to each of those two
            (all to one when the other side is outside), exactly, so totals are kept;
            refused when both sides are outside
  refused   when the link condition fails (the corners a and b share a neighbour
            corner, edge or triangle that is not around the edge: the collapse
            would glue the mesh into something that is not tetrahedra), or when any
            tetrahedron that keeps one of the two corners would end up flat or
            inverted (volume <= 0)
The outside counts as one extra corner for the link condition. The space of the
tetrahedra around a and b is kept: their outer surface does not move, and the
cone of a closed surface from any inner point has the same volume. kept (n',) are
the old rows of the remaining tetrahedra, in order; gone is the corner drawn into
the other (b after the choice above), left loose for drop_loose_corners().
"""
from __future__ import annotations

import numpy as np

from physics.cells.cell_volumes import cell_volumes
from physics.cells.tet_faces import tet_faces

_OUTSIDE = -1


def collapse_edge(corners: np.ndarray, tets: np.ndarray, values: np.ndarray, edge):
    """Mesh and values after edge (a, b) collapses into one corner, or None when refused."""
    faces, face_cells = tet_faces(tets)
    outer = faces[face_cells[:, 1] < 0]
    on_outside = np.zeros(len(corners), dtype=bool)
    on_outside[outer.ravel()] = True
    kept_corner, gone_corner = int(edge[0]), int(edge[1])
    if on_outside[kept_corner] and on_outside[gone_corner]:
        return None
    if on_outside[gone_corner]:
        kept_corner, gone_corner = gone_corner, kept_corner
    point = (corners[kept_corner] if on_outside[kept_corner]
             else (corners[kept_corner] + corners[gone_corner]) / 2)
    if not _link_holds(tets, outer, kept_corner, gone_corner):
        return None
    has_kept, has_gone = (tets == kept_corner).any(axis=1), (tets == gone_corner).any(axis=1)
    vanish = np.flatnonzero(has_kept & has_gone)
    moved_corners = corners.copy()
    moved_corners[kept_corner] = point
    moved_tets = np.where(tets == gone_corner, kept_corner, tets)
    touched = np.flatnonzero(has_kept ^ has_gone)
    if (cell_volumes(moved_corners, moved_tets[touched]) <= 0).any():
        return None
    handed = _handed_on(faces, face_cells, tets, vanish, kept_corner, gone_corner, values)
    if handed is None:
        return None
    kept = np.setdiff1d(np.arange(len(tets)), vanish)
    return moved_corners, moved_tets[kept], handed[kept], kept, gone_corner


def _handed_on(faces: np.ndarray, face_cells: np.ndarray, tets: np.ndarray, vanish: np.ndarray,
               kept_corner: int, gone_corner: int, values: np.ndarray):
    """Values after each vanishing tetrahedron hands its values to the cells across its two faces."""
    keys = np.sort(faces, axis=1)
    handed = values.astype(float).copy()
    for cell in vanish:
        others = [corner for corner in tets[cell] if corner not in (kept_corner, gone_corner)]
        across = []
        for end in (kept_corner, gone_corner):
            key = np.sort([end, *others])
            row = np.flatnonzero((keys == key).all(axis=1))[0]
            pair = face_cells[row]
            neighbour = int(pair[1] if pair[0] == cell else pair[0])
            if neighbour >= 0:
                across.append(neighbour)
        if not across:
            return None
        for neighbour in across:
            handed[neighbour] += handed[cell] / len(across)
        handed[cell] = 0.0
    return handed


def _link_holds(tets: np.ndarray, outer: np.ndarray, first: int, second: int) -> bool:
    """The link condition: what the two corners' links share is exactly the edge's link."""
    first_link = _link(tets, outer, (first,))
    second_link = _link(tets, outer, (second,))
    edge_link = _link(tets, outer, (first, second))
    return all(first_link[rank] & second_link[rank] == edge_link[rank] for rank in range(3))


def _link(tets: np.ndarray, outer: np.ndarray, simplex: tuple) -> list[set]:
    """The link of a corner or an edge as [corners, edges, triangles], the outside one extra corner."""
    pieces = [tuple(sorted(set(row) - set(simplex)))
              for row in tets[np.isin(tets, simplex).sum(axis=1) == len(simplex)].tolist()]
    pieces += [tuple(sorted((set(row) - set(simplex)) | {_OUTSIDE}))
               for row in outer[np.isin(outer, simplex).sum(axis=1) == len(simplex)].tolist()]
    ranks: list[set] = [set(), set(), set()]
    for piece in pieces:
        for size in range(1, len(piece) + 1):
            ranks[size - 1].update(_subsets(piece, size))
    return ranks


def _subsets(piece: tuple, size: int) -> set:
    """Every sorted subset of `piece` with `size` members."""
    if size == 1:
        return {(member,) for member in piece}
    if size == 2:
        return {(piece[one], piece[two]) for one in range(len(piece)) for two in range(one + 1, len(piece))}
    return {piece} if len(piece) == 3 else set()
