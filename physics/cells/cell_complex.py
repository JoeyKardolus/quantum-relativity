"""Cell complex: the topology of the woven cells, energy volumes and joint cells, each with its own corners.

Contract: a frozen record (built by joint_cells.py from the skeleton's tetrahedra) of
  keys          (k, 5) int: what each corner is, in skeleton corners (see joint_cells.py):
                (0, i, others of its tetrahedron sorted) an energy volume's own copy of
                skeleton corner i; (1, i, the face's other two sorted, -1) a point on a
                skeleton face near i; (2, i, j, -1, -1) a point on skeleton edge (i, j)
                near i; (3, i, -1, -1, -1) skeleton corner i itself (only on the outside)
  tets          (n, 4) int: each energy volume's own four corners, positively oriented,
                in the slot order of its skeleton tetrahedron
  faces         (f, 3) int: every triangle between two cells or on the outside, once,
                outward from its first cell
  face_cells    (f, 2) int: the triangle's first and second cell, -1 the outside
  face_joint    (f,) int: the joint whose joint cell is one side of the triangle, else -1
  cell_kind     (c,) int: KIND_VOLUME (0) energy volume, KIND_JOINT (1) joint cell,
                KIND_SEAM (2) seam cell, KIND_CORNER (3) corner cell; cells numbered
                energy volumes 0..n-1, then joint j as cell n + j, then seam cells (one
                per skeleton edge), then corner cells (one per skeleton corner)
  cell_skeleton (c, 4) int: the skeleton corners of each cell's place (its tetrahedron,
                its joint's face, its edge or its corner), -1 padded
  joints        (j, 2) int: the energy volumes of each joint (face_joints.py order)
  slot_joint    (n, 4) int: per skeleton tetrahedron and slot s, the joint on the face
                opposite s, -1 outside
  skeleton_tets (n, 4) int: the skeleton tetrahedra the cells were cut from
  skeleton_faces, skeleton_face_cells, skeleton_face_joint: tet_faces.py and
                face_joints.py of the skeleton tetrahedra
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KIND_VOLUME, KIND_JOINT, KIND_SEAM, KIND_CORNER = 0, 1, 2, 3


@dataclass(frozen=True)
class CellComplex:
    """Corners as skeleton keys, energy volumes' own tetrahedra, the triangles between cells, cell kinds."""

    keys: np.ndarray
    tets: np.ndarray
    faces: np.ndarray
    face_cells: np.ndarray
    face_joint: np.ndarray
    cell_kind: np.ndarray
    cell_skeleton: np.ndarray
    joints: np.ndarray
    slot_joint: np.ndarray
    skeleton_tets: np.ndarray
    skeleton_faces: np.ndarray
    skeleton_face_cells: np.ndarray
    skeleton_face_joint: np.ndarray
