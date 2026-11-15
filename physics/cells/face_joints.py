"""Face joints: the joints of a mesh, one per pair of cells that share at least one face.

Contract: `face_joints(face_cells, face_origin, face_area, thickness, rate)` returns
(joints, face_joint, joint_thickness, joint_rate, fresh):
  joints       (k, 2) int, a < b, sorted unique: every pair of real cells sharing a face
  face_joint   (f,) the row of joints each face is on, -1 for a face on the outside
  thickness, rate  (k,) carried from the old joints: face_origin[f] is the old joint
               the face was on before a topology change (-1: none, a new face); a
               joint takes the face-area-weighted mean of its faces' old thickness
               and rate (a joint left by a collapse, made of two old ones, takes
               their mean by area; a neighbour's face cut by a split keeps its joint's)
  fresh        (k,) True for a joint none of whose faces had an old joint (the tear
               of a volume split); its thickness and rate are 0 for the caller to set
A face with zero area counts with weight 1e-300 so its joint still gets its old values.
"""
from __future__ import annotations

import numpy as np


def face_joints(face_cells: np.ndarray, face_origin: np.ndarray, face_area: np.ndarray,
                thickness: np.ndarray, rate: np.ndarray):
    """Joints from the faces' cell pairs, with thickness and rate carried from the old joints."""
    inside = np.flatnonzero((face_cells >= 0).all(axis=1))
    ends = np.sort(face_cells[inside], axis=1)
    joints, slot = np.unique(ends, axis=0, return_inverse=True)
    slot = slot.ravel()
    face_joint = np.full(len(face_cells), -1, dtype=np.int64)
    face_joint[inside] = slot
    origin = face_origin[inside]
    known = origin >= 0
    weight = np.maximum(face_area[inside][known], 1e-300)
    count = len(joints)
    total = np.bincount(slot[known], weight, count)
    fresh = total <= 0
    joint_thickness = np.divide(np.bincount(slot[known], weight * thickness[origin[known]], count),
                                total, out=np.zeros(count), where=~fresh)
    joint_rate = np.divide(np.bincount(slot[known], weight * rate[origin[known]], count),
                           total, out=np.zeros(count), where=~fresh)
    return joints.astype(np.int64).reshape(-1, 2), face_joint, joint_thickness, joint_rate, fresh
