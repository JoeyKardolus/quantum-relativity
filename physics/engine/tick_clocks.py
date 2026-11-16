"""Tick clocks: every box's local clock advances by the proper time it lived this step.

Contract: `tick_clocks(state, settings)` adds step_time * clock_rate(quanta) to every
occupied box's clock (clock_rate.py: the more packed, joints included, the slower).
Empty boxes hold no time. The step counter is only the computer's counter; a box's clock
is its own time (bible: time is local). Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.clock_rate import clock_rate
from physics.engine.settings import Settings
from physics.engine.state import State


@wp.kernel
def _tick(quanta: wp.array3d(dtype=wp.int32), clock: wp.array3d(dtype=wp.float32), step_time: float,
          clock_packing: float):
    ti, tj, tk = wp.tid()
    amount = quanta[ti, tj, tk]
    if amount != 0:
        clock[ti, tj, tk] = clock[ti, tj, tk] + step_time * clock_rate(amount, clock_packing)


def tick_clocks(state: State, settings: Settings) -> None:
    """Advance every occupied box's clock by its own proper time for one step."""
    wp.launch(_tick, dim=state.shape, device=state.quanta.device,
              inputs=[state.quanta, state.clock, settings.step_time, settings.clock_packing])
