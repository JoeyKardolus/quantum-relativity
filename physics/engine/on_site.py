"""On site: each box's fluid phase turns by the energy per quantum it feels, for its own clock's time.

Contract: `on_site(state, settings, duration)` turns every occupied box's phase by
-fluid_potential * duration * rate (fluid_potential.py; rate its cell's clock rate,
state.rate, cell_laws.py): plus boxes by the pull, minus boxes by their joint's push,
every box also by the fluid's own pressure against its cell's mean packing. This is the
on-site half of the Strang splitting (step.py); it is an exact phase turn, so the
quanta are untouched (the turn itself is site_turn.py). Empty boxes are skipped. With
fluid substeps (substeps.py) a box turns for duration / its cell's own count, only while
`sub_index` is below that count, and after the first substep only in the cells whose
later substeps run here rather than cell by cell (later_substeps.py). Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.site_turn import site_turn
from physics.engine.settings import Settings
from physics.engine.state import State

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _on_site(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32),
             phase: wp.array3d(dtype=wp.float32), duration: float, rates: wp.array2d(dtype=wp.float32),
             pushes: wp.array2d(dtype=wp.float32), packings: wp.array2d(dtype=wp.float32), pressure: float, side: int,
             plans: wp.array2d(dtype=wp.int32), sub_index: int):
    ti, tj, tk = wp.tid()
    amount = quanta[ti, tj, tk]
    if amount == 0:
        return
    home = ti // side
    cell = owner[ti, tj, tk]
    plan = plans[home, cell]
    count = wp.abs(plan)
    if sub_index >= count or (sub_index > 0 and plan < 0):
        return
    span = duration / float(count)
    phase[ti, tj, tk] = site_turn(amount, pull[ti, tj, tk], pushes[home, cell], packings[home, cell], pressure,
                                  phase[ti, tj, tk], span, rates[home, cell])


def on_site(state: State, settings: Settings, duration: float, sub_index: int = 0) -> None:
    """Turn every occupied box's phase by its potential for `duration` (over its universe's substeps) of its own clock."""
    wp.launch(_on_site, dim=state.shape, device=state.quanta.device,
              inputs=[state.quanta, state.owner, state.pull, state.phase, float(duration), state.cells_d["rate"],
                      state.cells_d["push"], state.cells_d["packing"], float(settings.push), state.side, state.substeps,
                      int(sub_index)])
