"""Host boxes: one universe's per-box arrays copied from the GPU to numpy.

Contract: `host_boxes(state, universe=0)` returns a dict of numpy arrays of shape
(side, side, side), indexed [x, y, z] in that universe's own coordinates: quanta
(int32, signed whole numbers), phase, clock and pull (float32), owner (int32, the
extended-lattice cell each box belongs to, state.py), plus "cells_per_side" (int, 0
without a cell lattice) and "corners" (float32 (m + 1, m + 1, m + 1, 3), m =
cells_per_side + 2: the lattice vertices in box units). Copies once per call; the state
is not changed. Raises ValueError for a universe the batch does not hold.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import State


def host_boxes(state: State, universe: int = 0) -> dict:
    """quanta, phase, clock, pull and owner of one universe as host arrays, with its lattice."""
    rows = state.universe(universe)
    boxes = {name: np.ascontiguousarray(getattr(state, name)[rows.start:rows.stop].numpy())
             for name in ("quanta", "phase", "clock", "pull", "owner")}
    extent = state.corners.shape[1]
    boxes["corners"] = state.corners.numpy().reshape(state.universes, extent, extent, extent, 3)[universe].copy()
    boxes["cells_per_side"] = state.cells_per_side
    return boxes
