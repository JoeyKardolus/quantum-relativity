"""Tick clocks: every living cell's own clock moves on by one step at its own rate.

Contract: `tick_clocks(state, settings)` adds settings.step_time * rate to state.clock
for every living cell (rate from cell_laws.py: the more packed, the slower), the proper
time the cell has lived. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.settings import Settings
from physics.engine.state import State


@wp.kernel
def _tick(kind: wp.array(dtype=wp.int32), rate: wp.array(dtype=wp.float32), clock: wp.array(dtype=wp.float32),
          span: float):
    cell = wp.tid()
    if kind[cell] != 0:
        clock[cell] = clock[cell] + span * rate[cell]


def tick_clocks(state: State, settings: Settings) -> None:
    """Advance each cell's clock by step_time at its own rate."""
    wp.launch(_tick, dim=state.count, device=state.device,
              inputs=[state.kind, state.rate, state.clock, float(settings.step_time)])
