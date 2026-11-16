"""Host boxes: one universe's per-box arrays copied from the GPU to numpy.

Contract: `host_boxes(state, universe=0)` returns a dict of numpy arrays of shape
(side, side, side), indexed [x, y, z] in that universe's own coordinates: quanta
(int32, signed whole numbers), phase, clock and pull (float32). Copies once per call;
the state is not changed. Raises ValueError for a universe the batch does not hold.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import State


def host_boxes(state: State, universe: int = 0) -> dict:
    """quanta, phase, clock and pull of one universe as host arrays."""
    rows = state.universe(universe)
    return {name: np.ascontiguousarray(getattr(state, name)[rows.start:rows.stop].numpy())
            for name in ("quanta", "phase", "clock", "pull")}
