"""Mesh message: the mesh's topology for the page, sent only when it changes.

Contract: `mesh_message(engine)` returns a JSON-ready dict
  {"type": "mesh", "corner_count": k, "cells": n, "joints": j,
   "faces": [[corner, ...], ...]      each face's ring, outward from its first cell,
   "face_cells": [[a, b], ...]        each face's first and second cell, -1 outside,
   "face_joint": [j, ...]}            each face's joint, -1 outside
The corners' positions, pulls and pushes come per frame (frame_arrays.py) in the
same order: corner i, cell c and joint j here are row i, c and j there until the
next mesh message (the next collapse or split).
"""
from __future__ import annotations

from physics.cells.engine import Engine


def mesh_message(engine: Engine) -> dict:
    """Faces, their cells and joints, and the counts the frames will carry."""
    return {"type": "mesh", "corner_count": len(engine.corners),
            "cells": engine.grid.cell_count, "joints": len(engine.grid.joints),
            "faces": engine.faces.tolist(),
            "face_cells": engine.face_cells.tolist(), "face_joint": engine.face_joint.tolist()}
