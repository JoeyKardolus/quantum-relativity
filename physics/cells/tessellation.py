"""Tessellation: the exact, gap-free cell shapes of one power tessellation, as arrays.

Contract: a frozen record made by tessellate() for n working positions in a
periodic cube of side `side`:
  volume        (n,) each cell's tessellated volume; sums to side**3 (0 for an empty cell)
  centroid      (n, 3) each cell's centroid, in the same unwrapped image as its faces
                (may lie just outside the box); an empty cell's is its position
  face_cell     (f,) the cell each face belongs to
  face_other    (f,) the cell on the other side of that face
  face_area     (f,) the face's area
  corners       (m, 3) every face's corners in order, face after face, each in its
                own cell's unwrapped image; face i is corners[face_start[i]:
                face_start[i] + face_size[i]]. A face shared by cells a and b appears
                twice, once in each cell's list, as in any tessellation.
  face_start    (f,) where each face's corners begin in `corners`
  face_size     (f,) how many corners each face has
  pairs         (k, 2) int, a < b, every pair of cells sharing at least one face,
                sorted: the neighbour graph (and the joints) of this tessellation
  face_joint    (f,) the row of `pairs` each face is on, -1 for a face of a cell
                with itself (only in a box a few cells wide)
  pair_area     (k,) summed area of the faces of each pair, counted from the lower cell
  pair_gap      (k,) distance between the pair's two working positions (nearest image)
  seams         (m, 3) int, a < b < c, sorted unique: the three cells around each
                edge of the tessellation (an edge two faces of a cell share; one
                face to each of the other two cells). Edges whose three cells are
                not distinct (a cell meeting itself or one neighbour twice, only in
                a box a few cells wide) are left out.
Positions are labels for computing these shapes; no rule reads any of them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tessellation:
    """Volumes, centroids, faces, neighbour pairs and seams of one periodic power tessellation."""

    volume: np.ndarray
    centroid: np.ndarray
    face_cell: np.ndarray
    face_other: np.ndarray
    face_area: np.ndarray
    corners: np.ndarray
    face_start: np.ndarray
    face_size: np.ndarray
    pairs: np.ndarray
    face_joint: np.ndarray
    pair_area: np.ndarray
    pair_gap: np.ndarray
    seams: np.ndarray
