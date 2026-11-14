"""The "mesh" message: the current joint list, sent whenever it changes.

Contract: `mesh_message(simulation)` returns {"type": "mesh", "cells": n,
"pairs": [i0, j0, i1, j1, ...]}; joint k sits between energy volumes
pairs[2k] and pairs[2k + 1], in the same order as a frame's "squeeze" list.
"""
from __future__ import annotations

from physics.app.simulation import Simulation


def mesh_message(simulation: Simulation) -> dict:
    """The joint list of the simulation's current grid."""
    grid = simulation.grid
    return {"type": "mesh", "cells": int(grid.cell_count),
            "pairs": [int(cell) for cell in grid.joints.ravel()]}
