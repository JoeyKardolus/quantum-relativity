"""Spread pull: the one pull field goes one wave step on, from cell to cell.

Contract: `spread_pull(state, settings)` reads `pull`, `pull_old`, `packing`, `length`,
`cell_quanta`, `kind` and the neighbour tables, and writes `pull`, `pull_old`, the
working copy `pull_new` and `pull_made`. Nothing else.
Every living cell that holds energy takes one wave step (pull_wave_law.py) from what it
hears of its own neighbours (heard_pull.py) and its own source (pull_source.py: only
plus energy, by its packing; written to `pull_made`); `pull_old` becomes the pull as it
was before the step. No clock rate is in the step: the pull spreads between cells, and a
packed cell's slow clock does not make it pull less.
A cell reads its own neighbours and nothing else, so nothing here reaches further than
one neighbour in one step: no number is worked out over the whole grid. The level all
cells share is not held still, so it sinks faster every step (as the step squared); only differences of the pull are
ever read, and in float64 they keep.
An id that is no cell has pull 0. A living cell that holds nothing is empty space: it
makes no pull, carries none and takes no step, and no rule reads its value.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL
from physics.engine.axiom_5_packing.definitions.heard_pull import heard_pull
from physics.engine.axiom_5_packing.definitions.pull_source import pull_source
from physics.engine.axiom_5_packing.definitions.pull_wave_law import wave_step

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))


@wp.kernel
def _wave(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
          adj_at: wp.array(dtype=wp.int32), adj_n: wp.array(dtype=wp.int32),
          adj_id: wp.array(dtype=wp.int32), pull: wp.array(dtype=wp.float64),
          pull_old: wp.array(dtype=wp.float64), packing: wp.array(dtype=wp.float64),
          length: wp.array(dtype=wp.float32), light_speed: wp.float64,
          step_time: wp.float64, pull_strength: wp.float64, pull_new: wp.array(dtype=wp.float64),
          pull_made: wp.array(dtype=wp.float32)):
    """One thread per cell: its pull one wave step on, from its own neighbours and its own source."""
    cell = wp.tid()
    if kind[cell] == _NO_CELL:
        pull_new[cell] = wp.float64(0.0)
        return
    if cell_quanta[cell] <= wp.int64(0):
        pull_new[cell] = pull[cell]
        return
    heard = heard_pull(cell, kind, cell_quanta, adj_at, adj_n, adj_id, pull, length, light_speed, step_time)
    source = pull_source(kind[cell], packing[cell], pull_strength)
    pull_new[cell] = wave_step(pull[cell], pull_old[cell], heard[0], light_speed, step_time, source)
    pull_made[cell] = wp.float32(source)


def spread_pull(state, settings) -> None:
    """Take the pull of every cell that holds energy one wave step on."""
    wp.launch(_wave, dim=state.count,
              inputs=[state.kind, state.cell_quanta, state.adj_at, state.adj_n, state.adj_id, state.pull,
                      state.pull_old, state.packing, state.length, wp.float64(settings.light_speed),
                      wp.float64(settings.step_time), wp.float64(settings.pull_strength)],
              outputs=[state.pull_new, state.pull_made], device=state.device)
    wp.copy(state.pull_old, state.pull, count=state.count)
    wp.copy(state.pull, state.pull_new, count=state.count)
