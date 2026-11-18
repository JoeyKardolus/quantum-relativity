"""Start pull: the start's energy has always pulled, so the pull starts at rest, not at nothing.

Contract: `start_pull(state, settings)` sets every clock to 0 and writes `pull` and
`pull_old` (equal: the field stands still) as the pull at rest for the start's energy:
the field one wave step (`wave_rest`, pull_wave_law.py) leaves unchanged in every cell
up to one level shared by all, with mean 0. So no wave is switched on at step 1, and at
the start the pull is already deeper near more packed energy and weaker with distance.
It is found by sweeps: each cell that holds energy moves by what a wave step would add
to it, less the mean of that over all cells (in whole units of 2 ** -40, so the same on
any device), over its neighbours' mean weight. Sweeps stop when the largest move is
below a millionth of the pull's own spread, or after `40 * side ** 2`; the largest move
of the last sweep is kept in `state.pull_unrest` to be read. Needs `packing` and
`length` (so it runs after `size_cells` and `pack_cells`). Writes `pull`, `pull_old`,
`pull_new` (working copy), `clock`. Changes no quantum.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL
from physics.engine.axiom_5_packing.definitions.heard_pull import heard_pull
from physics.engine.axiom_5_packing.definitions.pull_source import pull_source
from physics.engine.axiom_5_packing.definitions.pull_wave_law import wave_rest

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))
_UNIT = wp.constant(wp.float64(2.0 ** -40))
_REST_SUM = 0
_CELLS = 1
_CHECK_EVERY = 100
_SWEEPS_PER_SIDE_SQUARED = 40
_STILL = 1e-6


@wp.kernel
def _unrest(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
            adj_at: wp.array(dtype=wp.int32), adj_n: wp.array(dtype=wp.int32), adj_id: wp.array(dtype=wp.int32),
            pull: wp.array(dtype=wp.float64), packing: wp.array(dtype=wp.float64),
            length: wp.array(dtype=wp.float32), light_speed: wp.float64, step_time: wp.float64,
            pull_strength: wp.float64, rest: wp.array(dtype=wp.float64), loud: wp.array(dtype=wp.float64),
            totals: wp.array(dtype=wp.int64)):
    """One thread per cell: what a wave step would add to its standing pull, and its neighbours' mean weight."""
    cell = wp.tid()
    rest[cell] = wp.float64(0.0)
    loud[cell] = wp.float64(0.0)
    if kind[cell] == _NO_CELL or cell_quanta[cell] <= wp.int64(0):
        return
    heard = heard_pull(cell, kind, cell_quanta, adj_at, adj_n, adj_id, pull, length, light_speed, step_time)
    source = pull_source(kind[cell], packing[cell], pull_strength)
    rest[cell] = wave_rest(heard[0], light_speed, step_time, source)
    loud[cell] = heard[1]
    wp.atomic_add(totals, _REST_SUM, wp.int64(wp.round(rest[cell] / _UNIT)))
    wp.atomic_add(totals, _CELLS, wp.int64(1))


@wp.kernel
def _settle(rest: wp.array(dtype=wp.float64), loud: wp.array(dtype=wp.float64), totals: wp.array(dtype=wp.int64),
            pull: wp.array(dtype=wp.float64), moved: wp.array(dtype=wp.float32)):
    """One thread per cell: move its pull by its unrest less the mean unrest, over its neighbours' weight."""
    cell = wp.tid()
    moved[cell] = wp.float32(0.0)
    if not (loud[cell] > wp.float64(0.0)):
        return
    mean = wp.float64(totals[_REST_SUM] / totals[_CELLS]) * _UNIT
    change = (rest[cell] - mean) / loud[cell]
    pull[cell] = pull[cell] + change
    moved[cell] = wp.float32(wp.abs(change))


def _sweep(state, settings, rest, loud, moved) -> None:
    """One sweep of every cell toward the pull at rest."""
    totals = wp.zeros(2, dtype=wp.int64, device=state.device)
    wp.launch(_unrest, dim=state.count,
              inputs=[state.kind, state.cell_quanta, state.adj_at, state.adj_n, state.adj_id, state.pull,
                      state.packing, state.length, wp.float64(settings.light_speed),
                      wp.float64(settings.step_time), wp.float64(settings.pull_strength)],
              outputs=[rest, loud, totals], device=state.device)
    wp.launch(_settle, dim=state.count, inputs=[rest, loud, totals], outputs=[state.pull, moved],
              device=state.device)


def _level_out(state) -> None:
    """Take the mean of the pull out, so the field at rest has mean 0 over the cells in use."""
    pull = state.pull.numpy()
    pull[:state.count] -= pull[:state.count].mean(dtype="float64")
    state.pull.assign(pull)


def start_pull(state, settings) -> None:
    """Set every clock to 0 and the pull to the field at rest for the energy as it lies."""
    state.clock.zero_()
    state.pull.zero_()
    rest = wp.zeros(state.capacity, dtype=wp.float64, device=state.device)
    loud = wp.zeros(state.capacity, dtype=wp.float64, device=state.device)
    moved = wp.zeros(state.capacity, dtype=wp.float32, device=state.device)
    state.pull_unrest = 0.0
    for sweep in range(_SWEEPS_PER_SIDE_SQUARED * state.side ** 2):
        _sweep(state, settings, rest, loud, moved)
        if (sweep + 1) % _CHECK_EVERY == 0:
            pull = state.pull.numpy()[:state.count]
            state.pull_unrest = float(moved.numpy()[:state.count].max())
            if state.pull_unrest <= _STILL * float(pull.max() - pull.min()):
                break
    _level_out(state)
    wp.copy(state.pull_old, state.pull)
