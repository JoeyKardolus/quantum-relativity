"""Cell laws: each cell's clock rate, the size its energy wants, its size if it is an energy volume, its push if it is a joint.

Contract: `cell_laws(state, settings)` reads state.cell_quanta and state.cell_boxes
(cell_count.py) and writes, per universe and cell id (float64 numpy):
  rate       1 / (1 + packing / clock_packing), packing = quanta / boxes (1 with local
             clocks off): everything in the cell runs on this clock (owner, 2026-11-16)
  wanted     (size_energy[k] / quanta) ** 3, k 0 for an energy volume and 1 for a joint:
             length = constant / energy (bible rule 4), so more energy wants less room
  size       for an energy volume its wanted size: its size follows its energy (bible
             rule 5: volumes do not vibrate on their own); a joint's size is its own,
             swinging (joint_swing.py)
  cell_push  for a joint push * (wanted / size) ** push_steepness, never below 0 (a joint
             only pushes; squeezed when smaller than its energy wants, steeply); 0 for an
             energy volume (it only pulls) and for no cell
Returns the squeeze per cell, (wanted / size) ** (1/3) - 1 for a joint (its wanted length
over its length, minus 1; 0 for any other id). No cell holds a size at rest.
"""
from __future__ import annotations

import numpy as np

from physics.engine.settings import Settings
from physics.engine.state import State


def cell_laws(state: State, settings: Settings) -> np.ndarray:
    """Clock rate, wanted size, volume size and joint push of every cell; returns each joint's squeeze."""
    amounts = state.cell_quanta.astype(np.float64)
    boxes = np.maximum(state.cell_boxes, 1).astype(np.float64)
    volumes = state.kind == 1
    joints = state.kind == -1
    rate = np.ones(amounts.shape)
    if settings.clock_packing > 0.0:
        rate = 1.0 / (1.0 + amounts / boxes / settings.clock_packing)
    constant = np.where(volumes, state.size_energy[:, :1], state.size_energy[:, 1:])
    wanted = np.where(amounts > 0, (constant / np.maximum(amounts, 1.0)) ** 3, 0.0)
    state.rate = rate
    state.wanted = wanted
    state.size = np.where(volumes, wanted, state.size)
    ratio = np.where(joints, wanted / np.maximum(state.size, 1e-6), 1.0)
    state.cell_push = np.where(joints, settings.push * ratio ** settings.push_steepness, 0.0)
    return np.where(joints, np.cbrt(ratio) - 1.0, 0.0)
