"""On site: each box's fluid phase turns by the energy per quantum it feels, for its own cell's clock time.

Contract: `on_site(state, settings, duration)` turns every box's phase in every living
cell by -fluid_potential * duration * rate (fluid_potential.py; rate the cell's clock
rate): a plus box by its cell's pull, every box by the fluid's own pressure against its
cell's mean per box (cell_quanta / 216). The on-site half of the Strang splitting
(step.py); an exact phase turn, so the quanta are untouched. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.settings import Settings
from physics.engine.site_turn import site_turn
from physics.engine.state import BLOCK, BOXES, State

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _on_site(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
             pull: wp.array(dtype=wp.float32), rate: wp.array(dtype=wp.float32),
             quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float32), duration: float,
             pressure: float, boxes: float):
    cell, one, two, three = wp.tid()
    sort = kind[cell]
    if sort == 0:
        return
    felt = float(0.0)
    if sort == 1:
        felt = pull[cell]
    phase[cell, one, two, three] = site_turn(quanta[cell, one, two, three], felt, float(cell_quanta[cell]) / boxes,
                                             pressure, phase[cell, one, two, three], duration, rate[cell])


def on_site(state: State, settings: Settings, duration: float) -> None:
    """Turn every box's phase by its potential for `duration` of its cell's own clock."""
    wp.launch(_on_site, dim=(state.count, BLOCK, BLOCK, BLOCK), device=state.device,
              inputs=[state.kind, state.cell_quanta, state.pull, state.rate, state.quanta, state.phase,
                      float(duration), float(settings.fluid_pressure), float(BOXES)])
