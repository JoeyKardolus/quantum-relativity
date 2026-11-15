"""Build fan: cut the mesh's faces into triangles and index their volume-gradient slots.

Contract: `build_fan(faces, face_cells, cell_count)` returns a Fan (fan.py). faces is a
sequence of int arrays (each face's corner ring, at least 3 corners), face_cells
(f, 2) each face's first cell and second cell (-1 outside). Raises ValueError for a
face with fewer than 3 corners.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from physics.cells.fan import Fan


def build_fan(faces: Sequence[np.ndarray], face_cells: np.ndarray, cell_count: int) -> Fan:
    """Fan triangles of every face and the (cell, corner) slots their gradients add into."""
    sizes = np.fromiter((len(ring) for ring in faces), dtype=np.int64, count=len(faces))
    if (sizes < 3).any():
        raise ValueError("every face needs at least 3 corners")
    flat = np.concatenate(faces).astype(np.int64) if len(faces) else np.zeros(0, np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    tri_face = np.repeat(np.arange(len(faces)), sizes - 2)
    within = np.arange(len(tri_face)) - np.repeat(np.cumsum(sizes - 2) - (sizes - 2), sizes - 2)
    base = starts[tri_face]
    triangles = np.stack([flat[base], flat[base + within + 1], flat[base + within + 2]], axis=1)
    sides = [(0, 1.0), (1, -1.0)]
    vertex, sign, cell = [], [], []
    for column, side_sign in sides:
        owner = face_cells[tri_face, column]
        real = np.flatnonzero(owner >= 0)
        positions = (real[:, None] * 3 + np.arange(3)).ravel()
        vertex.append(positions)
        sign.append(np.full(len(positions), side_sign))
        cell.append(np.repeat(owner[real], 3))
    entry_vertex = np.concatenate(vertex)
    entry_cell = np.concatenate(cell)
    corner_count = int(flat.max()) + 1 if len(flat) else 0
    keys = entry_cell * corner_count + triangles.ravel()[entry_vertex]
    unique_keys, entry_slot = np.unique(keys, return_inverse=True)
    return Fan(cell_count=cell_count, triangles=triangles, tri_face=tri_face,
               face_cells=np.asarray(face_cells, dtype=np.int64), entry_vertex=entry_vertex,
               entry_sign=np.concatenate(sign), entry_slot=entry_slot.ravel(),
               slot_cell=unique_keys // corner_count, slot_corner=unique_keys % corner_count)
