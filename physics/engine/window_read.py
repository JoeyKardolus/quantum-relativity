"""Window read: one box-shaped room of one universe's quanta, owners and marks, copied to the host.

Contract: `window_read(state, universe, low, extent)` returns (quanta, owner, marks), numpy
arrays of shape `extent` (three lengths) holding the boxes of universe `universe` from
coordinate `low` on each axis, taken round the side (cells_span.py gives a room that
needs it only in a universe that wraps). window_write.py puts such arrays back. A rule
that runs on the host (volume_split.py) reads and writes only the room it touches.
"""
from __future__ import annotations

import warp as wp

from physics.engine.state import State


@wp.kernel
def _read(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8),
          first: int, side: int, low: wp.vec3i, quanta_out: wp.array3d(dtype=wp.int32),
          owner_out: wp.array3d(dtype=wp.int32), marks_out: wp.array3d(dtype=wp.uint8)):
    ti, tj, tk = wp.tid()
    across = first + (low[0] + ti) % side
    up = (low[1] + tj) % side
    deep = (low[2] + tk) % side
    quanta_out[ti, tj, tk] = quanta[across, up, deep]
    owner_out[ti, tj, tk] = owner[across, up, deep]
    marks_out[ti, tj, tk] = marks[across, up, deep]


def window_read(state: State, universe: int, low, extent) -> tuple:
    """The room's quanta, owners and marks as host arrays."""
    device = state.quanta.device
    shape = tuple(int(value) for value in extent)
    quanta = wp.empty(shape, dtype=wp.int32, device=device)
    owner = wp.empty(shape, dtype=wp.int32, device=device)
    marks = wp.empty(shape, dtype=wp.uint8, device=device)
    wp.launch(_read, dim=shape, device=device,
              inputs=[state.quanta, state.owner, state.marks, universe * state.side, state.side,
                      wp.vec3i(*[int(value) for value in low]), quanta, owner, marks])
    return quanta.numpy(), owner.numpy(), marks.numpy()
