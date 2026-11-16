"""On site: each box's fluid phase turns by the energy per quantum it feels, for its own clock's time.

Contract: `on_site(state, settings, duration)` turns every occupied box's phase by
-box_potential * duration * clock_rate (box_potential.py, clock_rate.py): plus boxes by
the pull and their own fluid push, minus boxes by the joint's steep push. This is the
on-site half of the Strang splitting (step.py); it is an exact phase turn, so the
quanta are untouched. Empty boxes are skipped. With fluid substeps (substeps.py), a
universe u turns for duration / state.substeps[u] and only while `sub_index` is below
its count, so each universe takes its own substeps in one launch. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.box_potential import box_potential
from physics.engine.clock_rate import clock_rate
from physics.engine.settings import Settings
from physics.engine.state import State

PI = 3.14159265358979


@wp.kernel
def _on_site(quanta: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32),
             phase: wp.array3d(dtype=wp.float32), duration: float, volume_rest: float, volume_stiffness: float,
             joint_rest: float, push: float, push_steepness: float, clock_packing: float, side: int,
             counts: wp.array(dtype=wp.int32), sub_index: int):
    ti, tj, tk = wp.tid()
    amount = quanta[ti, tj, tk]
    if amount == 0:
        return
    count = counts[ti // side]
    if sub_index >= count:
        return
    span = duration / float(count)
    energy = box_potential(amount, pull[ti, tj, tk], volume_rest, volume_stiffness, joint_rest, push, push_steepness)
    angle = phase[ti, tj, tk] - energy * span * clock_rate(amount, clock_packing)
    phase[ti, tj, tk] = angle - 2.0 * PI * wp.floor((angle + PI) / (2.0 * PI))


def on_site(state: State, settings: Settings, duration: float, sub_index: int = 0) -> None:
    """Turn every occupied box's phase by its potential for `duration` (over its universe's substeps) of its own clock."""
    wp.launch(_on_site, dim=state.shape, device=state.quanta.device,
              inputs=[state.quanta, state.pull, state.phase, float(duration), settings.volume_rest,
                      settings.volume_stiffness, settings.joint_rest, settings.push, settings.push_steepness,
                      settings.clock_packing, state.side, state.substeps, int(sub_index)])
