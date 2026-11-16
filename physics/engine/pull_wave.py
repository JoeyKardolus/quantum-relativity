"""Pull wave: the pull spreads box to box at light speed, a wave sourced by plus quanta, on local clocks.

Contract: `pull_wave(state, settings)` advances the pull field one step of the wave
equation d2(pull)/dt2 = c**2 (laplacian(pull) - 4 pi G rho), c = light_speed,
G = pull_strength, rho = the plus quanta in the box (joints and empty boxes add none),
by leapfrog on the 7-box stencil, each box on its own cell's clock (owner, 2026-11-16:
everything in a cell, the pull too, runs on the cell's local clock): first pull_rate +=
step_time * rate * c**2 * (...), then pull += step_time * rate * pull_rate, rate the
box's cell's clock rate (state.rate; cell_laws.py runs first when the state has none).
Each box reads only its six face neighbours, so a change travels at most one box per
step. Its static limit is laplacian(pull) = 4 pi G rho, the pull falling as -G M / r.
Space ends where the numbers reach zero: an empty box, and every place beyond a wall,
holds pull 0. In a universe that wraps round (state.wrap) there is no wall to hold the
pull's level, so the pull's level over the whole universe (its mean, which turns every
plus box's phase alike and moves nothing) is taken out each step; that is a choice of
zero, not a force. Returns None.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.cell_count import cell_count
from physics.engine.cell_laws import cell_laws
from physics.engine.cell_upload import cell_upload
from physics.engine.face_step import face_step
from physics.engine.reach import reach
from physics.engine.settings import Settings
from physics.engine.state import State

PI = 3.14159265358979
LEVEL_UNIT = 1.0e9


@wp.kernel
def _speed_up(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32),
              rates: wp.array2d(dtype=wp.float32), pull: wp.array3d(dtype=wp.float32),
              pull_rate: wp.array3d(dtype=wp.float32), side: int, wrap: int, kick: float, source: float):
    ti, tj, tk = wp.tid()
    amount = quanta[ti, tj, tk]
    if amount == 0:
        pull_rate[ti, tj, tk] = 0.0
        return
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    centre = pull[ti, tj, tk]
    spread = -6.0 * centre
    for face in range(6):
        there, valid = reach(here, face_step(face), home, side, wrap)
        if valid:
            spread += pull[there[0], there[1], there[2]]
    density = float(wp.max(amount, 0))
    rate = rates[home, owner[ti, tj, tk]]
    pull_rate[ti, tj, tk] = pull_rate[ti, tj, tk] + kick * rate * (spread - source * density)


@wp.kernel
def _move(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32),
          rates: wp.array2d(dtype=wp.float32), pull_rate: wp.array3d(dtype=wp.float32),
          pull: wp.array3d(dtype=wp.float32), side: int, step_time: float):
    ti, tj, tk = wp.tid()
    if quanta[ti, tj, tk] == 0:
        pull[ti, tj, tk] = 0.0
        return
    rate = rates[ti // side, owner[ti, tj, tk]]
    pull[ti, tj, tk] = pull[ti, tj, tk] + step_time * rate * pull_rate[ti, tj, tk]


@wp.kernel
def _level(quanta: wp.array3d(dtype=wp.int32), field: wp.array3d(dtype=wp.float32), side: int,
           sums: wp.array2d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    if quanta[ti, tj, tk] == 0:
        return
    home = ti // side
    wp.atomic_add(sums, home, 0, wp.int64(wp.round(float(field[ti, tj, tk]) * LEVEL_UNIT)))
    wp.atomic_add(sums, home, 1, wp.int64(1))


@wp.kernel
def _lower(quanta: wp.array3d(dtype=wp.int32), field: wp.array3d(dtype=wp.float32), side: int,
           levels: wp.array(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    if quanta[ti, tj, tk] != 0:
        field[ti, tj, tk] = field[ti, tj, tk] - levels[ti // side]


def pull_wave(state: State, settings: Settings) -> None:
    """Advance the pull one leapfrog step of its wave equation on local clocks, neighbour boxes only."""
    if state.rate is None or state.rate.shape != state.kind.shape:
        cell_count(state)
        cell_laws(state, settings)
        cell_upload(state)
    if "rate" not in state.cells_d:
        cell_upload(state)
    kick = settings.step_time * settings.light_speed ** 2
    source = 4.0 * PI * settings.pull_strength
    device = state.quanta.device
    rates = state.cells_d["rate"]
    wp.launch(_speed_up, dim=state.shape, device=device,
              inputs=[state.quanta, state.owner, rates, state.pull, state.pull_rate, state.side, int(state.wrap),
                      float(kick), float(source)])
    wp.launch(_move, dim=state.shape, device=device,
              inputs=[state.quanta, state.owner, rates, state.pull_rate, state.pull, state.side, settings.step_time])
    if state.wrap:
        for field in (state.pull_rate, state.pull):
            _take_level(state, field)


def _take_level(state: State, field) -> None:
    """Subtract each universe's mean over its occupied boxes from `field`."""
    device = state.quanta.device
    sums = wp.zeros((state.universes, 2), dtype=wp.int64, device=device)
    wp.launch(_level, dim=state.shape, device=device, inputs=[state.quanta, field, state.side, sums])
    totals = sums.numpy()
    levels = (totals[:, 0] / LEVEL_UNIT / np.maximum(totals[:, 1], 1)).astype(np.float32)
    wp.launch(_lower, dim=state.shape, device=device,
              inputs=[state.quanta, field, state.side, wp.array(levels, dtype=wp.float32, device=device)])
