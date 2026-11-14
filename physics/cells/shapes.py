"""Shapes: the engine's exact cell shapes and joints, ready for the viewer to draw as they are.

Contract: `shapes(engine)` returns
  {"cells":  [{"id": i, "faces": [[x, y, z, x, y, z, ...], ...], "pull": p}, ...],
   "joints": [{"a": i, "b": j, "polygon": [x, y, z, ...], "push": q, "thickness": t}, ...]}
  cells    one entry per energy volume: every face of its tessellated cell, each a flat
           list of its corners in order, all in one unwrapped periodic image of that
           cell (corners may lie just outside [0, side)); an empty cell has no faces.
  pull     how hard that energy volume pulls: its energy per volume, scaled 0..1 over
           the grid ((packing - min) / (max - min); 0 everywhere on an even grid).
  joints   one entry per shared face: a < b, the face as cell a has it (in a's image).
           A pair that shares two faces (across the periodic wrap, only in a box a few
           cells wide) is one joint drawn as both faces. Every entry is a joint of the
           grid and every joint of the grid has at least one entry.
  push     how hard that joint pushes: its squeeze toward its collapse limit, 0..1
           (joint_squeeze.py).
  thickness  the joint's volume / the area of its faces: how thick a slab it is, capped
           at a quarter of a typical cell's width so tiny faces do not draw as streaks.
Plain Python floats and ints, ready for JSON. Reads the engine only.
"""
from __future__ import annotations

import numpy as np

from physics.cells.engine import Engine

#: Drawn slab thickness is capped at this share of a typical cell's width: on a tiny face,
#: volume / area explodes and the slab would shoot across the view as a streak.
_THICKEST = 0.25


def shapes(engine: Engine) -> dict:
    """Every cell's faces with its pull, every joint's face with its push and thickness."""
    tessellation, grid = engine.tessellation, engine.grid
    packing = grid.energy / grid.volume
    spread = packing.max() - packing.min()
    pull = (packing - packing.min()) / spread if spread > 0 else np.zeros(grid.cell_count)
    numbers = tessellation.corners.ravel().tolist()
    flat = [numbers[3 * start:3 * (start + size)] for start, size
            in zip(tessellation.face_start.tolist(), tessellation.face_size.tolist())]
    faces = [[] for _ in range(grid.cell_count)]
    for cell, polygon in zip(tessellation.face_cell.tolist(), flat):
        faces[cell].append(polygon)
    cells = [{"id": cell, "faces": faces[cell], "pull": float(pull[cell])}
             for cell in range(grid.cell_count)]
    thickness = np.minimum(grid.joint_volume / np.maximum(tessellation.pair_area, 1e-300),
                           _THICKEST * np.cbrt(grid.volume.mean()))
    joints = []
    lower = (tessellation.face_joint >= 0) & (tessellation.face_cell < tessellation.face_other)
    for face in np.flatnonzero(lower).tolist():
        joint = int(tessellation.face_joint[face])
        joints.append({"a": int(grid.joints[joint, 0]), "b": int(grid.joints[joint, 1]),
                       "polygon": flat[face], "push": float(grid.squeeze[joint]),
                       "thickness": float(thickness[joint])})
    return {"cells": cells, "joints": joints}
