"""Pull wave: the pull spreads box to box at light speed, a wave sourced by plus quanta (no all-space solve).

Contract: `pull_wave(state, settings)` advances the pull field one step of the wave
equation d2(pull)/dt2 = c**2 (laplacian(pull) - 4 pi G rho), c = light_speed,
G = pull_strength, rho = the plus quanta in the box (joints and empty boxes add none),
by leapfrog on the 7-box stencil: first pull_rate += step_time * c**2 * (...), then
pull += step_time * pull_rate. Each box reads only its six face neighbours, so a change
travels at most one box per step and the pull of a source arrives at distance r after
r / c (pull_arrival test). Its static limit is laplacian(pull) = 4 pi G rho, the pull
falling as -G M / r. Space ends where the numbers reach zero: an empty box, and every
place beyond the grid's walls, holds pull 0 (fixed), so the pull spreads through space
only. The pull runs on the counter's time, not on local clocks. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.face_step import face_step
from physics.engine.inside import inside
from physics.engine.settings import Settings
from physics.engine.state import State

PI = 3.14159265358979


@wp.kernel
def _speed_up(quanta: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32),
              pull_rate: wp.array3d(dtype=wp.float32), side: int, kick: float, source: float):
    ti, tj, tk = wp.tid()
    amount = quanta[ti, tj, tk]
    if amount == 0:
        pull_rate[ti, tj, tk] = 0.0
        return
    here = wp.vec3i(ti, tj, tk)
    centre = pull[ti, tj, tk]
    spread = -6.0 * centre
    for face in range(6):
        there = here + face_step(face)
        if inside(there, ti // side, side):
            spread += pull[there[0], there[1], there[2]]
    density = float(wp.max(amount, 0))
    pull_rate[ti, tj, tk] = pull_rate[ti, tj, tk] + kick * (spread - source * density)


@wp.kernel
def _move(quanta: wp.array3d(dtype=wp.int32), pull_rate: wp.array3d(dtype=wp.float32),
          pull: wp.array3d(dtype=wp.float32), step_time: float):
    ti, tj, tk = wp.tid()
    if quanta[ti, tj, tk] == 0:
        pull[ti, tj, tk] = 0.0
        return
    pull[ti, tj, tk] = pull[ti, tj, tk] + step_time * pull_rate[ti, tj, tk]


def pull_wave(state: State, settings: Settings) -> None:
    """Advance the pull one leapfrog step of its wave equation, neighbour boxes only."""
    kick = settings.step_time * settings.light_speed ** 2
    source = 4.0 * PI * settings.pull_strength
    shape = state.shape
    device = state.quanta.device
    wp.launch(_speed_up, dim=shape, device=device,
              inputs=[state.quanta, state.pull, state.pull_rate, state.side, float(kick), float(source)])
    wp.launch(_move, dim=shape, device=device,
              inputs=[state.quanta, state.pull_rate, state.pull, settings.step_time])
