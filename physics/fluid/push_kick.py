"""Push kick: B's pressure changes the B flow over one substep.

Contract: `push_kick(grid, settings, dt)` adds (P_low * t_low - P_high * t_high) * dt
per axis to b_momentum in every box, P_low and P_high the pressure on its two faces
along that axis and t their face clocks. A face between two boxes carries the smaller
of their pressures (b_pressure.py) and the mean of their clock rates (clock_rate.py:
each box's own time); an outer wall face carries the inside box's own pressure (the
wall pushes back) and clock. Face-based, so every push one box gets its neighbour gets
back: momentum is kept inside the grid. The smaller pressure keeps an almost empty box beside a packed one
from being kicked at all (it has no push of its own to receive), so the edge of the
energy expands outward from its own pressure without the empty space flying off.
First-order where pressure is smooth. In place, after which transport.py moves the
energy by the new velocities (symplectic Euler, stable at the CFL substep).
"""
from __future__ import annotations

import warp as wp

from physics.fluid.b_pressure import b_pressure
from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.settings import Settings

_QUANTUM = wp.constant(QUANTUM)


@wp.func
def _pressure_at(b_energy: wp.array3d(dtype=wp.int64), i: int, j: int, k: int, b_rest: float, push: float,
                 power: float, collapse: float) -> float:
    return b_pressure(float(wp.max(b_energy[i, j, k], wp.int64(0))) * _QUANTUM, b_rest, push, power, collapse)


@wp.func
def _face_pressure(b_energy: wp.array3d(dtype=wp.int64), i: int, j: int, k: int, axis: int, towards: int,
                   side: int, own: float, b_rest: float, push: float, power: float, collapse: float) -> float:
    other = wp.vec3i(i, j, k)
    other[axis] = other[axis] + towards
    if other[axis] < 0 or other[axis] >= side:
        return own
    return wp.min(own, _pressure_at(b_energy, other[0], other[1], other[2], b_rest, push, power, collapse))


@wp.func
def _face_clock(clock: wp.array3d(dtype=float), i: int, j: int, k: int, axis: int, towards: int,
                side: int) -> float:
    other = wp.vec3i(i, j, k)
    other[axis] = other[axis] + towards
    if other[axis] < 0 or other[axis] >= side:
        return clock[i, j, k]
    return 0.5 * (clock[i, j, k] + clock[other[0], other[1], other[2]])


@wp.kernel
def _push(b_energy: wp.array3d(dtype=wp.int64), clock: wp.array3d(dtype=float),
          b_momentum: wp.array3d(dtype=wp.vec3), side: int,
          b_rest: float, push: float, power: float, collapse: float, dt: float):
    i, j, k = wp.tid()
    own = _pressure_at(b_energy, i, j, k, b_rest, push, power, collapse)
    force = wp.vec3(0.0, 0.0, 0.0)
    for axis in range(3):
        high = _face_pressure(b_energy, i, j, k, axis, 1, side, own, b_rest, push, power, collapse)
        low = _face_pressure(b_energy, i, j, k, axis, -1, side, own, b_rest, push, power, collapse)
        force[axis] = (low * _face_clock(clock, i, j, k, axis, -1, side)
                       - high * _face_clock(clock, i, j, k, axis, 1, side))
    b_momentum[i, j, k] = b_momentum[i, j, k] + force * dt


def push_kick(grid: Grid, settings: Settings, dt: float) -> None:
    """Accelerate the B flow by its pressure differences for dt."""
    wp.launch(_push, dim=(grid.side,) * 3, device=grid.b_energy.device,
              inputs=[grid.b_energy, grid.clock, grid.b_momentum, grid.side, settings.b_rest, settings.push,
                      settings.push_power, settings.collapse_compression, float(dt)])
