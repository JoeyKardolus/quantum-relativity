"""The binary frame: display position, size and colour value per volume, nothing else.

Contract: `page_frame(simulation, mesh_id)` returns bytes, little-endian:
  header    HEADER_WORDS uint32: FRAME_MAGIC, step, mesh_id, count, 0, 0, 0, 0
  position  float32 count * 3 (x, y, z per volume, the state's volume order), from
            `simulation.positions`, the drawing layout (never physics)
  size      float32 count, the state's size per volume (the page draws a ball of it)
  colour    uint8 count, `colour_ranks`: rank of pull for an energy volume, rank of
            squeeze for a joint, 0..255
The page applies it to the mesh with the same mesh_id and drops it otherwise. 16 bytes
plus one per volume; no JSON, no contacts.
"""
from __future__ import annotations

import numpy as np

from physics.app.colour_ranks import colour_ranks
from physics.app.simulation import Simulation

FRAME_MAGIC = 0x31415251  # "QRA1" read as little-endian uint32
HEADER_WORDS = 8


def page_frame(simulation: Simulation, mesh_id: int) -> bytes:
    """One binary frame of the simulation's current state, tagged with `mesh_id`."""
    state = simulation.state
    positions = np.ascontiguousarray(simulation.positions, dtype="<f4").reshape(-1, 3)
    sizes = np.ascontiguousarray(state.size, dtype="<f4").ravel()
    colours = colour_ranks(state)
    header = np.array([FRAME_MAGIC, simulation.steps, mesh_id, len(positions), 0, 0, 0, 0], dtype="<u4")
    return header.tobytes() + positions.tobytes() + sizes.tobytes() + colours.tobytes()
