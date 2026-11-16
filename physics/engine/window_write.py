"""Window write: a box-shaped room of quanta, owners and marks put back into one universe's boxes.

Contract: `window_write(state, universe, low, quanta, owner, marks)` writes the three host
arrays (one shape, as window_read.py gave them) over the boxes of universe `universe`
from coordinate `low` on each axis, taken round the side. Every box of the room is
written; the caller changed only the boxes its rule may change. Returns None.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.state import State


@wp.kernel
def _write(quanta_in: wp.array3d(dtype=wp.int32), owner_in: wp.array3d(dtype=wp.int32),
           marks_in: wp.array3d(dtype=wp.uint8), first: int, side: int, low: wp.vec3i,
           quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8)):
    ti, tj, tk = wp.tid()
    across = first + (low[0] + ti) % side
    up = (low[1] + tj) % side
    deep = (low[2] + tk) % side
    quanta[across, up, deep] = quanta_in[ti, tj, tk]
    owner[across, up, deep] = owner_in[ti, tj, tk]
    marks[across, up, deep] = marks_in[ti, tj, tk]


def window_write(state: State, universe: int, low, quanta: np.ndarray, owner: np.ndarray, marks: np.ndarray) -> None:
    """Write the room's host arrays back over the universe's boxes."""
    device = state.quanta.device
    wp.launch(_write, dim=quanta.shape, device=device,
              inputs=[wp.array(quanta, dtype=wp.int32, device=device), wp.array(owner, dtype=wp.int32, device=device),
                      wp.array(marks, dtype=wp.uint8, device=device), universe * state.side, state.side,
                      wp.vec3i(*[int(value) for value in low]), state.quanta, state.owner, state.marks])
