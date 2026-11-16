"""Tick clocks: every box's clock and every cell's clock advance by the proper time they lived this step.

Contract: `tick_clocks(state, settings)` adds step_time * rate to every occupied box's
clock, rate its cell's clock rate (state.rate, cell_laws.py: the more packed the cell,
joints included, the slower), and step_time * rate to every cell's own clock
(state.cell_clock). Empty boxes hold no time. The step counter is only the computer's
counter; a cell's clock is its own time (bible: time is local). Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.settings import Settings
from physics.engine.state import State


@wp.kernel
def _tick(owner: wp.array3d(dtype=wp.int32), rates: wp.array2d(dtype=wp.float32), clock: wp.array3d(dtype=wp.float32),
          side: int, step_time: float):
    ti, tj, tk = wp.tid()
    cell = owner[ti, tj, tk]
    if cell >= 0:
        clock[ti, tj, tk] = clock[ti, tj, tk] + step_time * rates[ti // side, cell]


def tick_clocks(state: State, settings: Settings) -> None:
    """Advance every occupied box's and every cell's clock by its own proper time for one step."""
    wp.launch(_tick, dim=state.shape, device=state.quanta.device,
              inputs=[state.owner, state.cells_d["rate"], state.clock, state.side, settings.step_time])
    state.cell_clock = state.cell_clock + settings.step_time * state.rate * (state.kind != 0)
