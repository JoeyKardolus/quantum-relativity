"""Substeps: how many equal fluid substeps each universe takes this step, set by its steepest face.

Contract: `substeps(state, settings)` reads, per universe, the steepest face: the largest
|turn rate difference| between two face-neighbouring boxes of one cell (the hops act
only there, hop.py), a box's turn rate being box_potential * its cell's clock rate
(on_site.py).
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
from physics.engine.face_step import face_step
from physics.engine.reach import reach
from physics.engine.settings import Settings
from physics.engine.state import State


@wp.kernel
def _steepest(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32),
              rates: wp.array2d(dtype=wp.float32), pushes: wp.array2d(dtype=wp.float32), side: int, wrap: int,
              steepest: wp.array(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    held = quanta[ti, tj, tk]
    if held == 0:
        return
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    cell = owner[ti, tj, tk]
    own = box_potential(held, pull[ti, tj, tk], pushes[home, cell]) * rates[home, cell]
    largest = float(0.0)
    for face in range(3):
        there, valid = reach(here, face_step(2 * face), home, side, wrap)
        if valid and owner[there[0], there[1], there[2]] == cell:
            other = box_potential(quanta[there[0], there[1], there[2]], pull[there[0], there[1], there[2]],
                                  pushes[home, cell]) * rates[home, cell]
            largest = wp.max(largest, wp.abs(other - own))
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
              inputs=[state.quanta, state.owner, state.pull, state.cells_d["rate"], state.cells_d["push"], state.side,
                      int(state.wrap), steepest])
    needed = np.array([max(1, math.ceil(float(value) * settings.step_time / settings.max_turn))
                       for value in steepest.numpy()], dtype=np.int64)
    state.capped += needed > settings.max_substeps
    counts = np.minimum(needed, settings.max_substeps).astype(np.int32)
    state.substeps.assign(counts)
    return counts
