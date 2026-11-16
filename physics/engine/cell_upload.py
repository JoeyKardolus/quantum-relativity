"""Cell upload: the cell table the box kernels read, copied to the GPU.

Contract: `cell_upload(state)` copies the host cell table (state.py, cell_laws.py) to the
state's device and returns a dict of Warp arrays, each (universes, capacity): kind
(int32), between (vec2i), size, rate and push (float32; rate 1, push 0 before
cell_laws has run). The box kernels read a box's cell through its owner. Called again
after anything on the host changes the table.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.state import State


def cell_upload(state: State) -> dict:
    """The cell table as Warp arrays on the state's device."""
    device = state.quanta.device
    shape = state.kind.shape
    rate = state.rate if state.rate is not None and state.rate.shape == shape else np.ones(shape)
    push = state.cell_push if state.cell_push is not None and state.cell_push.shape == shape else np.zeros(shape)
    state.cells_d = {
        "kind": wp.array(state.kind.astype(np.int32), dtype=wp.int32, device=device),
        "between": wp.array(np.ascontiguousarray(state.between.astype(np.int32)), dtype=wp.vec2i, device=device),
        "size": wp.array(state.size.astype(np.float32), dtype=wp.float32, device=device),
        "rate": wp.array(rate.astype(np.float32), dtype=wp.float32, device=device),
        "push": wp.array(push.astype(np.float32), dtype=wp.float32, device=device),
    }
    return state.cells_d
