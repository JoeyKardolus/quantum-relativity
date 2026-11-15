"""Joint carry: after a topology change, each joint takes the thickness and rate of the joint on its face before.

Contract: `joint_carry(old_faces, old_face_joint, thickness, rate, faces, face_joint,
joint_count)` returns (thickness, rate, fresh) per new joint. old_faces (f0, 3) are
the faces before, with their corners already renamed onto the corners they became
(a corner drawn into another by a joint collapse takes that one's number);
old_face_joint their joints (-1 outside); faces, face_joint the new ones. A new
joint whose face (as a set of corners) was a joint's face before takes its
thickness and rate; when two old joint faces became the same face (the two faces of
a tetrahedron that fell in by a collapse), it takes their mean. fresh (k,) is True
for a joint whose face was no joint before (the tear of a volume split); its
thickness and rate are 0 for the caller to set.
"""
from __future__ import annotations

import numpy as np


def joint_carry(old_faces: np.ndarray, old_face_joint: np.ndarray, thickness: np.ndarray,
                rate: np.ndarray, faces: np.ndarray, face_joint: np.ndarray, joint_count: int):
    """New joints' thickness and rate from the old joints on the same faces; which are fresh."""
    width = int(max(old_faces.max(initial=0), faces.max(initial=0))) + 1
    was_joint = old_face_joint >= 0
    old_keys = _keys(old_faces[was_joint], width)
    old_joint = old_face_joint[was_joint]
    inside = face_joint >= 0
    new_keys = _keys(faces[inside], width)
    order = np.argsort(old_keys, kind="stable")
    sorted_keys = old_keys[order]
    low = np.searchsorted(sorted_keys, new_keys, side="left")
    high = np.searchsorted(sorted_keys, new_keys, side="right")
    found = high - low
    rows = np.repeat(np.arange(len(new_keys)), found)
    matches = order[np.concatenate([np.arange(start, stop) for start, stop in zip(low, high)])
                    .astype(np.int64)] if len(rows) else np.zeros(0, dtype=np.int64)
    target = face_joint[inside][rows]
    total = np.bincount(target, minlength=joint_count).astype(float)
    fresh = total == 0
    carried_thickness = np.divide(np.bincount(target, thickness[old_joint[matches]], joint_count), total,
                                  out=np.zeros(joint_count), where=~fresh)
    carried_rate = np.divide(np.bincount(target, rate[old_joint[matches]], joint_count), total,
                             out=np.zeros(joint_count), where=~fresh)
    return carried_thickness, carried_rate, fresh


def _keys(faces: np.ndarray, width: int) -> np.ndarray:
    """One int per face, the same for the same set of three corners."""
    ends = np.sort(faces, axis=1).astype(np.int64)
    return (ends[:, 0] * width + ends[:, 1]) * width + ends[:, 2]
