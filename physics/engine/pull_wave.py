"""Pull wave: the pull spreads from cell to cell at light speed over the sizes of the cells it crosses.

Contract: `pull_wave(state, settings)` advances state.pull by one step of the wave
equation on the cells' relations (leapfrog; state.pull_old is the pull a step ago).
A cell knows only its neighbours and its distance to each, half its own length plus
half the neighbour's, d = (length + neighbour length) / 2 (the sum of the sizes between
their middles). With N neighbours the Laplacian in three dimensions is
6 / N * sum((neighbour pull - pull) / d ** 2), so per step
  pull' = 2 pull - pull_old + rate ** 2 * (sum(w * (neighbour pull - pull)) / N
                                           - (light_speed * step_time) ** 2 * source)
  w      = min(6 * (light_speed * step_time / d) ** 2, 1)
  source = 4 pi G * plus quanta / length ** 3 for an energy volume, 0 for a joint
`rate` is the cell's local clock (a packed cell passes the pull on slower). A change
moves one neighbour per step at most (exactly: beyond that reach nothing changes), and
its bulk travels at light speed over the sizes of the cells it crosses; at rest it is
the static pull of the plus quanta, weaker with distance, deeper round a more packed
(more energy, smaller) volume. Joints carry it and add none. The cap on w keeps the
step stable for any cell size (cells shorter than a step of light pass the pull on
slower, never faster). The lattice wraps, so the pull's mean has no meaning: the mean
change is taken out every step (summed in whole 2 ** -40 units, the same in any order),
which shifts every cell alike and changes no difference. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.neighbour import neighbour
from physics.engine.settings import Settings
from physics.engine.state import State

wp.set_module_options({"fuse_fp": False})

_UNIT = 1099511627776.0  # numerics: 2**40 per unit, whole-number sums


@wp.kernel
def _wave(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64), length: wp.array(dtype=wp.float32),
          rate: wp.array(dtype=wp.float32), pull: wp.array(dtype=wp.float32), pull_old: wp.array(dtype=wp.float32),
          pull_new: wp.array(dtype=wp.float32), adj_at: wp.array(dtype=wp.int32), adj_n: wp.array(dtype=wp.int32),
          adj_pool: wp.array(dtype=wp.int32), side: int, light_step: float, strength: float,
          sums: wp.array(dtype=wp.int64)):
    cell = wp.tid()
    sort = kind[cell]
    if sort == 0:
        return
    count = adj_n[cell]
    own = length[cell]
    here = pull[cell]
    total = float(0.0)
    for index in range(count):
        other = neighbour(cell, index, side, adj_at, adj_pool)
        distance = 0.5 * (own + length[other])
        total += wp.min(6.0 * light_step * light_step / (distance * distance), 1.0) * (pull[other] - here)
    source = float(0.0)
    if sort == 1:
        source = strength * float(cell_quanta[cell]) / (own * own * own)
    pace = rate[cell]
    after = 2.0 * here - pull_old[cell] + pace * pace * (total / float(count) - light_step * light_step * source)
    pull_new[cell] = after
    wp.atomic_add(sums, 0, wp.int64(wp.float64(after - here) * wp.float64(_UNIT)))
    wp.atomic_add(sums, 1, wp.int64(1))


@wp.kernel
def _settle(kind: wp.array(dtype=wp.int32), pull: wp.array(dtype=wp.float32), pull_old: wp.array(dtype=wp.float32),
            sums: wp.array(dtype=wp.int64)):
    cell = wp.tid()
    if kind[cell] == 0:
        return
    shift = wp.float32(wp.float64(sums[0]) / wp.float64(_UNIT) / wp.float64(sums[1]))
    pull_old[cell] = pull[cell] + shift


def pull_wave(state: State, settings: Settings) -> None:
    """Advance the pull one step on the cells' relations, at light speed over their sizes."""
    sums = wp.zeros(2, dtype=wp.int64, device=state.device)
    wp.launch(_wave, dim=state.count, device=state.device,
              inputs=[state.kind, state.cell_quanta, state.length, state.rate, state.pull, state.pull_old,
                      state.pull_new, state.adj_at, state.adj_n, state.adj_pool, state.side,
                      float(settings.light_speed * settings.step_time),
                      float(4.0 * 3.14159265358979 * settings.pull_strength), sums])
    wp.launch(_settle, dim=state.count, device=state.device, inputs=[state.kind, state.pull, state.pull_old, sums])
    state.pull_old, state.pull, state.pull_new = state.pull_old, state.pull_new, state.pull
