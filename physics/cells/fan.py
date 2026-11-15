"""Fan: the mesh's faces cut into triangles, indexed for measuring volumes and their gradients.

Contract: a frozen record made by build_fan() from the mesh topology (engine.py), rebuilt
only when the topology changes (start, joint collapse, volume split):
  cell_count    number of cells n
  triangles     (t, 3) corner indices: each face's ring v0 .. v(m-1) cut into the fan
                (v0, vi, vi+1), i = 1 .. m-2, in the ring's order (outward from the
                face's first cell)
  tri_face      (t,) the face each triangle belongs to
  face_cells    (f, 2) each face's first cell and its second (-1: outside the cube)
  entry_vertex  (e,) for every triangle corner seen from each real cell of its face,
                its position in triangles.ravel()
  entry_sign    (e,) +1 for the face's first cell, -1 for its second
  entry_slot    (e,) the (cell, corner) slot the entry adds to
  slot_cell     (s,) per slot its cell
  slot_corner   (s,) per slot its corner
Both cells of a face use the same triangles, so the cells' volumes fill the mesh with
no gaps and no overlaps whatever the corners' positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fan:
    """Triangles of every face and the (cell, corner) slots of their volume gradients."""

    cell_count: int
    triangles: np.ndarray
    tri_face: np.ndarray
    face_cells: np.ndarray
    entry_vertex: np.ndarray
    entry_sign: np.ndarray
    entry_slot: np.ndarray
    slot_cell: np.ndarray
    slot_corner: np.ndarray
