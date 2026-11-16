"""Host boxes: one universe's per-box arrays and cell graph copied from the GPU to numpy.

Contract: `host_boxes(state, universe=0)` returns a dict of numpy arrays: quanta (int32,
signed whole numbers), phase, clock and pull (float32) and owner (int32, the cell each
box belongs to, -1 for empty space), each (side, side, side) indexed [x, y, z] in that
universe's own coordinates; and the universe's cell graph per cell id (state.py): kind,
between, place, size, speed, cell_clock, size_energy, and "wrap" (whether the universe
wraps round at its edges). The graph part is what State.from_host takes as `lattice`.
Copies once per call; the state is not changed. Raises ValueError for a universe the
batch does not hold.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import State


def host_boxes(state: State, universe: int = 0) -> dict:
    """quanta, phase, clock, pull and owner of one universe as host arrays, with its cell graph."""
    rows = state.universe(universe)
    boxes = {name: np.ascontiguousarray(getattr(state, name)[rows.start:rows.stop].numpy())
             for name in ("quanta", "phase", "clock", "pull", "owner")}
    for name in ("kind", "between", "place", "size", "speed", "cell_clock"):
        boxes[name] = np.array(getattr(state, name)[universe])
    boxes["size_energy"] = np.array(state.size_energy[universe])
    boxes["wrap"] = bool(state.wrap)
    return boxes
