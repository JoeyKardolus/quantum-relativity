"""Flow velocity: how fast the energy of one kind flows through a box.

Contract: a Warp function, usable inside kernels only. `flow_velocity(momentum,
quanta)` returns momentum / energy, energy = quanta * QUANTUM, for a box holding more
than VELOCITY_FLOOR energy; a box holding less (empty open space, or a few quanta
left by rounding) has no defined flow and returns zero, so it adds no wave to the
time step and carries no momentum of its own across a face.
"""
from __future__ import annotations

import warp as wp

from physics.fluid.grid import QUANTUM

VELOCITY_FLOOR = 1e-6
_QUANTUM = wp.constant(QUANTUM)
_FLOOR = wp.constant(VELOCITY_FLOOR)


@wp.func
def flow_velocity(momentum: wp.vec3, quanta: wp.int64) -> wp.vec3:
    """Momentum over energy, or zero in a box too empty to have a flow."""
    energy = float(quanta) * _QUANTUM
    if energy <= _FLOOR:
        return wp.vec3(0.0, 0.0, 0.0)
    return momentum / energy
