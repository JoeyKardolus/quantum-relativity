"""Tick clocks: each cell's own clock goes on by the local time this step is worth.

Contract: `tick_clocks(state, settings)` adds `step_time * rate` to `clock` of every
living cell that holds energy. It is the first thing a step does after clearing the
ledgers, with the rate every other rule of this step uses, so `clock` is the local time
the cell has lived through, a ledger no rule reads. A cell that holds nothing
is empty space and does nothing: its clock stands. Reads `kind`, `cell_quanta`, `rate`.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))


@wp.kernel
def _tick(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64), rate: wp.array(dtype=wp.float32),
          step_time: wp.float64, clock: wp.array(dtype=wp.float64)):
    cell = wp.tid()
    if kind[cell] != _NO_CELL and cell_quanta[cell] > wp.int64(0):
        clock[cell] = clock[cell] + step_time * wp.float64(rate[cell])


def tick_clocks(state, settings) -> None:
    """Add one step of local time to the clock of every living cell that holds energy."""
    wp.launch(_tick, dim=state.count,
              inputs=[state.kind, state.cell_quanta, state.rate, wp.float64(settings.step_time)],
              outputs=[state.clock], device=state.device)
