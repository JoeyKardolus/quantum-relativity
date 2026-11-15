"""Fastest wave: the largest per-box wave-speed sum, which sets the stable substep (CFL).

Contract: `fastest_wave(grid, settings)` returns, as a host float, the maximum over
boxes of sum over the three axes of max(|u_A|, |u_B| + c_B) along that axis, with
u_A, u_B the flow velocities (flow_velocity.py) and c_B the B wave speed
(b_wave_speed.py). A substep of courant / fastest_wave keeps every flux a box sends
out within its content (donor cell, transport.py). Raises
ValueError when a momentum is not finite (the state broke; nothing is taken).
"""
from __future__ import annotations

import math

import warp as wp

from physics.fluid.b_wave_speed import b_wave_speed
from physics.fluid.flow_velocity import flow_velocity
from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.settings import Settings

_QUANTUM = wp.constant(QUANTUM)
_WORK: dict[str, wp.array] = {}


@wp.kernel
def _row_fastest(a_energy: wp.array3d(dtype=wp.int64), b_energy: wp.array3d(dtype=wp.int64),
                 a_momentum: wp.array3d(dtype=wp.vec3), b_momentum: wp.array3d(dtype=wp.vec3),
                 side: int, b_rest: float, push: float, power: float, collapse: float,
                 fastest: wp.array(dtype=float)):
    """Each thread scans one row along z and raises the maximum once."""
    i, j = wp.tid()
    row = float(0.0)
    for k in range(side):
        a_velocity = flow_velocity(a_momentum[i, j, k], a_energy[i, j, k])
        b_velocity = flow_velocity(b_momentum[i, j, k], b_energy[i, j, k])
        sound = b_wave_speed(float(b_energy[i, j, k]) * _QUANTUM, b_rest, push, power, collapse)
        total = float(0.0)
        for axis in range(3):
            total += wp.max(wp.abs(a_velocity[axis]), wp.abs(b_velocity[axis]) + sound)
        if not wp.isfinite(total):
            total = 1.0e30
        row = wp.max(row, total)
    wp.atomic_max(fastest, 0, row)


def fastest_wave(grid: Grid, settings: Settings) -> float:
    """The largest sum of wave speeds over the axes in any box."""
    device = str(grid.a_energy.device)
    if device not in _WORK:
        _WORK[device] = wp.zeros(1, dtype=float, device=device)
    fastest = _WORK[device]
    fastest.zero_()
    wp.launch(_row_fastest, dim=(grid.side, grid.side), device=device,
              inputs=[grid.a_energy, grid.b_energy, grid.a_momentum, grid.b_momentum, grid.side,
                      settings.b_rest, settings.push, settings.push_power, settings.collapse_compression,
                      fastest])
    speed = float(fastest.numpy()[0])
    if not math.isfinite(speed) or speed >= 1.0e30:
        raise ValueError("a flow velocity stopped being finite")
    return speed
