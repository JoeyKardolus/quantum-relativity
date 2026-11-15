"""Fastest wave: the largest per-box wave-speed sum, which sets the stable substep (CFL).

Contract: `fastest_wave(grid, settings)` returns, as a host float, the maximum over
boxes of the box's clock rate (grid.clock, clock_rate.py: the box's own time per unit
of step time) times the sum over the three axes of max(s_A, s_B + c_B) along that
axis plus settings.leak where its A counts (the leak gives at most leak per unit of
own time, leak.py), with
s_A, s_B the sending speeds (sending_speed.py: |u| of the flow velocity,
flow_velocity.py, but 0 into a closed wall) and c_B the B wave speed
(b_wave_speed.py), where a kind holding less than floor = settings.cfl_floor *
grid.start_a_mean energy in a box adds nothing for that box (its term is 0). A
substep of courant / fastest_wave keeps every flux a kind above the floor sends out
within its content (donor cell, transport.py). The clock is the point of local time:
a packed core runs slowly, so its fast inner flow no longer sets every box's substep.
A kind below the floor is near-empty vapour; its outflow is capped at its content instead (transport.py). Measured with
every box counted: on the earlier ball start (seed 1, side 64, unevenness 0.5) from
step 18 the substep was set by A vapour of about 1e-6 per box (just above the
velocity floor) falling at 20 to 65 boxes per time unit into the sealed core, 100
substeps per step; with the floor, 10 (the core itself, about 4). On the filled
start (side 128, unevenness 0.3) from step 14, by A vapour of 1e-6 at 130 to 580 in
boxes whose B was not vapour: hence the floor per kind. Raises ValueError when
a momentum is not finite in any box (the state broke; nothing is taken).
"""
from __future__ import annotations

import math

import warp as wp

from physics.fluid.b_wave_speed import b_wave_speed
from physics.fluid.flow_velocity import flow_velocity
from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.sending_speed import sending_speed
from physics.fluid.settings import Settings

_QUANTUM = wp.constant(QUANTUM)
_WORK: dict[str, wp.array] = {}


@wp.kernel
def _row_fastest(a_energy: wp.array3d(dtype=wp.int64), b_energy: wp.array3d(dtype=wp.int64),
                 a_momentum: wp.array3d(dtype=wp.vec3), b_momentum: wp.array3d(dtype=wp.vec3),
                 clock: wp.array3d(dtype=float), side: int, b_rest: float, push: float, power: float,
                 collapse: float, floor: float, leak: float, fastest: wp.array(dtype=float)):
    """Each thread scans one row along z and raises the maximum once."""
    i, j = wp.tid()
    row = float(0.0)
    for k in range(side):
        a_velocity = flow_velocity(a_momentum[i, j, k], a_energy[i, j, k])
        b_velocity = flow_velocity(b_momentum[i, j, k], b_energy[i, j, k])
        sound = b_wave_speed(float(b_energy[i, j, k]) * _QUANTUM, b_rest, push, power, collapse)
        a_counts = float(a_energy[i, j, k]) >= floor
        b_counts = float(b_energy[i, j, k]) >= floor
        total = float(0.0)
        place = wp.vec3i(i, j, k)
        for axis in range(3):
            a_wave = sending_speed(a_velocity, axis, place[axis], side)
            b_wave = sending_speed(b_velocity, axis, place[axis], side) + sound
            if not wp.isfinite(a_wave) or not wp.isfinite(b_wave):
                total = 1.0e30
            else:
                if not a_counts:
                    a_wave = 0.0
                if not b_counts:
                    b_wave = 0.0
                total += wp.max(a_wave, b_wave)
        if a_counts:
            total += leak
        row = wp.max(row, clock[i, j, k] * total)
    wp.atomic_max(fastest, 0, row)


def fastest_wave(grid: Grid, settings: Settings) -> float:
    """The largest clock-weighted sum of wave speeds over the axes in any box above the CFL floor."""
    device = str(grid.a_energy.device)
    if device not in _WORK:
        _WORK[device] = wp.zeros(1, dtype=float, device=device)
    fastest = _WORK[device]
    fastest.zero_()
    wp.launch(_row_fastest, dim=(grid.side, grid.side), device=device,
              inputs=[grid.a_energy, grid.b_energy, grid.a_momentum, grid.b_momentum, grid.clock, grid.side,
                      settings.b_rest, settings.push, settings.push_power, settings.collapse_compression,
                      settings.cfl_floor * grid.start_a_mean / QUANTUM, settings.leak, fastest])
    speed = float(fastest.numpy()[0])
    if not math.isfinite(speed) or speed >= 1.0e30:
        raise ValueError("a flow velocity stopped being finite")
    return speed
