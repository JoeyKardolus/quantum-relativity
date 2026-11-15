"""Mesh message: the topology of the woven cells for the page, sent only when it changes.

Contract: `mesh_message(engine)` returns a JSON-ready dict
  {"type": "mesh", "corner_count": k, "cells": c, "energy_volumes": n, "joints": j,
   "faces": [[corner, ...], ...]      each triangle's ring, outward from its first cell,
   "face_cells": [[a, b], ...]        each triangle's first and second cell, -1 outside,
   "face_joint": [j, ...],            the joint whose joint cell is a side of it, else -1
   "cell_kind": [kind, ...]}          per cell 0 energy volume, 1 joint cell, 2 seam
                                      cell, 3 corner cell (2 and 3 are joint material
                                      too, where joints meet along a seam or at a corner)
c counts every cell (energy volumes 0..n-1 first, then joint j as cell n + j). The
corners' positions, pulls and pushes come per frame (frame_arrays.py) in the same
order: corner i, cell c and joint j here are row i, c and j there until the next mesh
message (the next collapse or split).
"""
from __future__ import annotations

from physics.cells.engine import Engine


def mesh_message(engine: Engine) -> dict:
    """Faces, their cells and joints, each cell's kind, and the counts the frames will carry."""
    return {"type": "mesh", "corner_count": len(engine.corners), "cells": len(engine.cell_kind),
            "energy_volumes": engine.grid.cell_count, "joints": len(engine.grid.joints),
            "faces": engine.faces.tolist(), "face_cells": engine.face_cells.tolist(),
            "face_joint": engine.face_joint.tolist(), "cell_kind": engine.cell_kind.tolist()}
