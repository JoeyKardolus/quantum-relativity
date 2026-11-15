"""The binary frame: corner positions, pull per cell and push per joint, nothing else.

Contract: `page_frame(simulation, mesh_id)` returns bytes, little-endian:
  header   HEADER_WORDS uint32: FRAME_MAGIC, step, mesh_id, corner_count,
           cell_count, joint_count, 0, 0
  corners  float32 corner_count * 3 (x, y, z per corner, the mesh's corner order)
  pull     uint8 cell_count, 0..255 (harder pull is higher)
  push     uint8 joint_count, 0..255 (harder push is higher)
straight from `physics.cells.frame_arrays(engine)`. The page applies it to the mesh
with the same mesh_id and drops it otherwise. About 12 bytes per corner plus one per
cell and joint; no JSON, no faces.
"""
from __future__ import annotations

import numpy as np

from physics.app.simulation import Simulation
from physics.cells import frame_arrays

FRAME_MAGIC = 0x31465251  # "QRF1" read as little-endian uint32
HEADER_WORDS = 8


def page_frame(simulation: Simulation, mesh_id: int) -> bytes:
    """One binary frame of the simulation's current engine, tagged with `mesh_id`."""
    corners, pull, push = frame_arrays(simulation.engine)
    corners = np.ascontiguousarray(corners, dtype="<f4").reshape(-1, 3)
    pull = np.ascontiguousarray(pull, dtype=np.uint8).ravel()
    push = np.ascontiguousarray(push, dtype=np.uint8).ravel()
    header = np.array([FRAME_MAGIC, simulation.steps, mesh_id, len(corners), len(pull), len(push), 0, 0],
                      dtype="<u4")
    return header.tobytes() + corners.tobytes() + pull.tobytes() + push.tobytes()
