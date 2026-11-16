"""Reach (Warp func): the box an offset away from a box, in the reading box's own universe, round its edges when it wraps.

Contract: `reach(box, offset, home, side, wrap)` returns (there, valid): the stacked-array
index of the box `offset` (a vec3i) away from `box`, in universe `home` (first indices
home * side .. home * side + side - 1, state.py). With wrap = 1 the universe wraps round
(opposite faces meet) and every offset is valid; with wrap = 0 an index past the
universe's walls is not valid (no space: no hop, no pull, no owner change across them;
`there` is then clamped inside and must not be used). No rule ever reads a neighbouring
universe.
"""
from __future__ import annotations

import warp as wp


@wp.func
def reach(box: wp.vec3i, offset: wp.vec3i, home: int, side: int, wrap: int):
    """The box `offset` away inside universe `home`, and whether it is space of that universe."""
    local = wp.vec3i(box[0] - home * side + offset[0], box[1] + offset[1], box[2] + offset[2])
    valid = True
    for axis in range(3):
        if local[axis] < 0 or local[axis] >= side:
            if wrap == 1:
                local[axis] = (local[axis] + side) % side
            else:
                valid = False
                local[axis] = wp.clamp(local[axis], 0, side - 1)
    return wp.vec3i(home * side + local[0], local[1], local[2]), valid
