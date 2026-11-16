"""On site: each box's fluid phase turns by the energy per quantum it feels, for its own clock's time.

Contract: `on_site(state, settings, duration)` turns every occupied box's phase by
-fluid_potential * duration * rate (fluid_potential.py; rate its cell's clock rate,
state.rate, cell_laws.py): plus boxes by the pull, minus boxes by their joint's push,
every box also by the fluid's own pressure against its cell's mean packing. This is the
on-site half of the Strang splitting (step.py); it is an exact phase turn, so the
quanta are untouched. Empty boxes are skipped. With fluid substeps (substeps.py), a
universe u turns for duration / state.substeps[u] and only while `sub_index` is below
its count, so each universe takes its own substeps in one launch. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.fluid_potential import fluid_potential
from physics.engine.settings import Settings
from physics.engine.state import State

PI = 3.14159265358979


@wp.kernel
def _on_site(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32),
             phase: wp.array3d(dtype=wp.float32), duration: float, rates: wp.array2d(dtype=wp.float32),
             pushes: wp.array2d(dtype=wp.float32), packings: wp.array2d(dtype=wp.float32), pressure: float, side: int,
             counts: wp.array(dtype=wp.int32), sub_index: int):
    ti, tj, tk = wp.tid()
    amount = quanta[ti, tj, tk]
    if amount == 0:
        return
    home = ti // side
    count = counts[home]
    if sub_index >= count:
        return
    span = duration / float(count)
    cell = owner[ti, tj, tk]
    energy = fluid_potential(amount, pull[ti, tj, tk], pushes[home, cell], packings[home, cell], pressure)
    angle = phase[ti, tj, tk] - energy * span * rates[home, cell]
    phase[ti, tj, tk] = angle - 2.0 * PI * wp.floor((angle + PI) / (2.0 * PI))


def on_site(state: State, settings: Settings, duration: float, sub_index: int = 0) -> None:
    """Turn every occupied box's phase by its potential for `duration` (over its universe's substeps) of its own clock."""
    wp.launch(_on_site, dim=state.shape, device=state.quanta.device,
              inputs=[state.quanta, state.owner, state.pull, state.phase, float(duration), state.cells_d["rate"],
                      state.cells_d["push"], state.cells_d["packing"], float(settings.push), state.side, state.substeps,
                      int(sub_index)])
