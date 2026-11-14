"""The "mesh" message: the current connection list, sent whenever it changes.

Contract: `mesh_message(simulation)` returns {"type": "mesh", "joints": n,
"pairs": [i0, j0, i1, j1, ...]}; connection c joins joints pairs[2c] and
pairs[2c + 1], in the same order as a frame's "energy" list.
"""
from __future__ import annotations

from physics.app.simulation import Simulation


def mesh_message(simulation: Simulation) -> dict:
    """The connection list of the simulation's current network."""
    network = simulation.network
    return {"type": "mesh", "joints": network.joint_count,
            "pairs": network.pairs.ravel().tolist()}
