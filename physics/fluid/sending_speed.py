"""Sending speed: how fast a box can send energy along one axis, nothing through a closed wall.

Contract: a Warp function, usable inside kernels only. `sending_speed(velocity, axis,
place, side)` returns |velocity[axis]|, or 0 when that component points through the
grid's outer face beside the box at index `place` along that axis (place 0 and
velocity < 0, or place side - 1 and velocity > 0): the walls are closed (transport.py),
so a flow into a wall sends nothing. Measured: on the filled start (seed 1, side 128,
unevenness 0.3) from step 21 the substep was set by A thrown from the core into the
grid's corners and edges, sitting on the wall at 130 to 150 boxes per time unit
into it, moving nowhere.
"""
from __future__ import annotations

import warp as wp


@wp.func
def sending_speed(velocity: wp.vec3, axis: int, place: int, side: int) -> float:
    """|velocity| along axis, or 0 into the closed wall beside the box."""
    component = velocity[axis]
    if (place == 0 and component < 0.0) or (place == side - 1 and component > 0.0):
        return 0.0
    return wp.abs(component)
