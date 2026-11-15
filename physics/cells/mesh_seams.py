"""Mesh seams: the seams of the mesh, the edges where three joint walls meet.

Contract: `mesh_seams(faces, face_cells)` returns (m, 3) int, a < b < c, sorted unique:
the three energy volumes around every mesh edge (two consecutive corners of a face
ring) that lies on exactly three faces, all between real cells, pairwise (a, b),
(a, c), (b, c). Edges on the cube's outside, edges inside one cell's crease (two
faces with the same neighbour, left by a joint collapse) and edges where more than
three faces meet are not seams. Every pair of a seam is a joint (a shared face).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def mesh_seams(faces: Sequence[np.ndarray], face_cells: np.ndarray) -> np.ndarray:
    """Per three-face edge between three distinct real cells, those three cells sorted."""
    sizes = np.fromiter((len(ring) for ring in faces), dtype=np.int64, count=len(faces))
    flat = np.concatenate(faces).astype(np.int64)
    following = np.concatenate([np.roll(ring, -1) for ring in faces]).astype(np.int64)
    edge_face = np.repeat(np.arange(len(faces)), sizes)
    low, high = np.minimum(flat, following), np.maximum(flat, following)
    keys = low * (int(flat.max()) + 1) + high
    order = np.argsort(keys, kind="stable")
    keys, edge_face = keys[order], edge_face[order]
    _, start, size = np.unique(keys, return_index=True, return_counts=True)
    three = start[size == 3]
    trio = edge_face[three[:, None] + np.arange(3)]
    sides = face_cells[trio]
    real = (sides >= 0).all(axis=(1, 2))
    sides = np.sort(sides[real], axis=2)
    pairs = np.sort(sides[:, :, 0] * (int(face_cells.max()) + 1) + sides[:, :, 1], axis=1)
    distinct = (pairs[:, 0] != pairs[:, 1]) & (pairs[:, 1] != pairs[:, 2])
    cells = np.sort(sides[distinct].reshape(-1, 6), axis=1)
    triangle = ((cells[:, 0] == cells[:, 1]) & (cells[:, 2] == cells[:, 3])
                & (cells[:, 4] == cells[:, 5]) & (cells[:, 1] != cells[:, 2])
                & (cells[:, 3] != cells[:, 4]))
    seams = cells[triangle][:, [0, 2, 4]]
    return np.unique(seams, axis=0).reshape(-1, 3).astype(np.int64)
