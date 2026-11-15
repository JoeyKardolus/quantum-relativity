"""Cell sizes: the measured volume of every cell of the woven cells, from its triangles.

Contract: `cell_sizes(corners, faces, face_cells, cell_count)` returns (c,) per cell
its signed volume: the sum over the triangles on it, each turned outward from the
cell (faces are outward from their first cell, face_cells), of
(a - r) . ((b - r) x (c - r)) / 6, r one corner of the cell (rounding stays small
for small cells far from the origin). Positive while the cell is the right way out.
Every inner triangle is counted once outward for each of its two cells, so the sizes
sum to the volume inside the outer triangles: no gaps, no overlaps.
"""
from __future__ import annotations

import numpy as np


def cell_sizes(corners: np.ndarray, faces: np.ndarray, face_cells: np.ndarray, cell_count: int) -> np.ndarray:
    """Per cell its signed volume from the triangles on it."""
    cell, ring = _sides(faces, face_cells)
    anchor = np.zeros(cell_count, dtype=np.int64)
    anchor[cell[::-1]] = ring[::-1, 0]
    reference = corners[anchor[cell]]
    first, second, third = (corners[ring[:, column]] - reference for column in range(3))
    terms = np.einsum("ij,ij->i", first, np.cross(second, third)) / 6
    return np.bincount(cell, terms, cell_count)


def _sides(faces: np.ndarray, face_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every (cell, triangle turned outward from it) pair, the outside left out."""
    first, second = face_cells[:, 0], face_cells[:, 1]
    inner = second >= 0
    cell = np.concatenate([first, second[inner]])
    ring = np.concatenate([faces, faces[inner][:, ::-1]])
    return cell, ring
