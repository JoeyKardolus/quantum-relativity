"""Cell targets: the size each cell of the woven cells aims at.

Contract: `cell_targets(rest, thickness, corners, faces, face_cells, cell_kind, built)`
returns (c,) per cell (cell_complex.py numbering):
  energy volume  rest, its own size from its energy and quanta (rest_volume.py, rule 4)
  joint cell     its thickness times its area, the mean area of its two big sides (the
                 triangles it shares with its two energy volumes, face_areas.py): the
                 joint's own thickness, swinging (swing_joints.py), sets its volume
  seam, corner   built, their size where place_corners.py puts their corners for the
                 joints' thicknesses now: joint material whose size follows the joints
                 that meet there (cell_sizes.py of those corners)
Only sizes are asked for; the cells' shapes follow their neighbours freely.
"""
from __future__ import annotations

import numpy as np

from physics.cells.cell_complex import KIND_JOINT, KIND_VOLUME
from physics.cells.face_areas import face_areas


def cell_targets(rest: np.ndarray, thickness: np.ndarray, corners: np.ndarray, faces: np.ndarray,
                 face_cells: np.ndarray, cell_kind: np.ndarray, built: np.ndarray) -> np.ndarray:
    """Per cell its target size: rest for energy volumes, thickness times area for joints, built otherwise."""
    count = int((cell_kind == KIND_VOLUME).sum())
    first, second = face_cells[:, 0], face_cells[:, 1]
    big_side = (first < count) & (second >= 0) & (cell_kind[np.maximum(second, 0)] == KIND_JOINT)
    area = np.bincount(second[big_side] - count, face_areas(corners, faces[big_side]), len(thickness)) / 2
    targets = np.array(built, dtype=float)
    targets[:count] = rest
    targets[count:count + len(thickness)] = thickness * area
    return targets
