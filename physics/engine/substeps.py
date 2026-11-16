"""Substeps: how many equal fluid substeps each universe takes this step, set by its steepest face.

Contract: `substeps(state, settings)` reads, per universe, the steepest face: the largest
|turn rate difference| between two face-neighbouring boxes of one sign (the hops act
only there, hop.py), a box's turn rate being box_potential * clock_rate (on_site.py).
Over one substep of length step_time / n that face's phase difference turns by
steepest * step_time / n; n is the smallest whole number keeping it at or below
settings.max_turn, at least 1 and at most settings.max_substeps. Numerics, not a rule
(owner 2026-11-16, decision E): the steep push turned a squeezed joint's phase by
radians a step, the split fluid step then scrambled the phases, and collapse ran away;
cutting the step where it is steep integrates the push stably at any squeeze, and the
step's counter time stays step_time. Each universe gets its own count, so a universe
steps the same alone as in a batch. Writes the counts into state.substeps, adds 1 to
state.capped for each universe that needed more than max_substeps, and returns the
counts (int32 numpy, one per universe). settings.max_turn = 0 gives 1 everywhere (the
coarse step) without reading the boxes.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from physics.engine.box_potential import box_potential
from physics.engine.clock_rate import clock_rate
from physics.engine.face_step import face_step
from physics.engine.inside import inside
from physics.engine.settings import Settings
from physics.engine.state import State


@wp.func
def _rate(amount: int, field: float, volume_rest: float, volume_stiffness: float, joint_rest: float, push: float,
          push_steepness: float, clock_packing: float) -> float:
    """How fast a box's phase turns: its potential on its own clock."""
    return (box_potential(amount, field, volume_rest, volume_stiffness, joint_rest, push, push_steepness)
            * clock_rate(amount, clock_packing))


@wp.kernel
def _steepest(quanta: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32), side: int,
              volume_rest: float, volume_stiffness: float, joint_rest: float, push: float, push_steepness: float,
              clock_packing: float, steepest: wp.array(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    held = quanta[ti, tj, tk]
    if held == 0:
        return
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    own = _rate(held, pull[ti, tj, tk], volume_rest, volume_stiffness, joint_rest, push, push_steepness,
                clock_packing)
    largest = float(0.0)
    for face in range(3):
        there = here + face_step(2 * face)
        if inside(there, home, side):
            other = quanta[there[0], there[1], there[2]]
            if other != 0 and (other > 0) == (held > 0):
                rate = _rate(other, pull[there[0], there[1], there[2]], volume_rest, volume_stiffness, joint_rest,
                             push, push_steepness, clock_packing)
                largest = wp.max(largest, wp.abs(rate - own))
    if largest > 0.0:
        wp.atomic_max(steepest, home, largest)


def substeps(state: State, settings: Settings) -> np.ndarray:
    """Per universe the substep count its steepest same-sign face needs, written to state.substeps."""
    if settings.max_turn <= 0.0:
        counts = np.ones(state.universes, dtype=np.int32)
        state.substeps.assign(counts)
        return counts
    device = state.quanta.device
    steepest = wp.zeros(state.universes, dtype=wp.float32, device=device)
    wp.launch(_steepest, dim=state.shape, device=device,
              inputs=[state.quanta, state.pull, state.side, settings.volume_rest, settings.volume_stiffness,
                      settings.joint_rest, settings.push, settings.push_steepness, settings.clock_packing, steepest])
    needed = np.array([max(1, math.ceil(float(value) * settings.step_time / settings.max_turn))
                       for value in steepest.numpy()], dtype=np.int64)
    state.capped += needed > settings.max_substeps
    counts = np.minimum(needed, settings.max_substeps).astype(np.int32)
    state.substeps.assign(counts)
    return counts
