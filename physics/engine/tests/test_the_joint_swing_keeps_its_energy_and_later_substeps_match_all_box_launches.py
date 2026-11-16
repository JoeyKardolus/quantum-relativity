"""Numerics that must not change what the rules say (2026-11-17, the engine made to run far bigger universes).

1. The joint swing is solved on the GPU (joint_swing.py). It is frictionless (axiom 4:
   the only friction is energy crossing a joint), so one step must keep the swing energy
   E = 1/2 v^T M^+ v + sum_i U_i(s_i), U_i(s) = push * wanted_i^6 / (5 s^5) - load_i * s
   (M the clock-weighted graph Laplacian over joints sharing a volume, over the inertia),
   and the joints' summed size. Measured on new_universe(8, 10, seed 1) after 20 steps
   of the whole engine (so the joints are moving), one more swing with the pull as load:
   the energy and the summed size change by less than 1e-9 of their size.
2. Each cell takes its own fluid substeps and the substeps after the first run only over
   the listed boxes of the cells that need them (substeps.py, later_substeps.py). The
   fluid never leaves its cell, so this must give, to the last bit, what launches over
   all boxes give. Staged: new_universe(4, 10, seed 3) with one plus box raised to 30
   times its quanta (a steep face in one volume), 5 steps each way.
"""
from __future__ import annotations

import sys

import numpy as np

from physics.engine import Settings, cell_count, cell_laws, joint_swing, new_universe, step
from physics.engine.tests._gpu import needs_gpu


def _swing_energy(state, settings: Settings, load: np.ndarray) -> float:
    joints = np.flatnonzero((state.kind[0] == -1) & (state.wanted[0] > 0.0))
    index = {int(joint): place for place, joint in enumerate(joints)}
    laplacian = np.zeros((joints.size, joints.size))
    for volume in np.flatnonzero(state.kind[0] == 1):
        members = [index[int(joint)] for joint in state.joints_of(0, int(volume)) if int(joint) in index]
        for first in members:
            for second in members:
                if first != second:
                    weight = state.rate[0, joints[first]] * state.rate[0, joints[second]] / settings.joint_inertia
                    laplacian[first, second] -= weight
                    laplacian[first, first] += weight
    speed = state.speed[0, joints]
    size = state.size[0, joints]
    moving = 0.5 * speed @ np.linalg.pinv(laplacian, hermitian=True) @ speed
    held = settings.push * state.wanted[0, joints] ** 6 / (5.0 * size ** 5) - load[0, joints] * size
    return float(moving + held.sum())


@needs_gpu
def test_one_swing_keeps_the_swing_energy_and_the_summed_size():
    settings = Settings()
    state = new_universe(8, 10, seed=1)
    for _ in range(20):
        step(state, settings)
    load = cell_count(state)
    cell_laws(state, settings)
    joints = state.kind[0] == -1
    before = _swing_energy(state, settings, load)
    summed = float(state.size[0, joints].sum())
    assert np.abs(state.speed[0, joints]).max() > 1e-3
    pressed = joint_swing(state, settings, load)
    after = _swing_energy(state, settings, load)
    print(f"swing energy {before:.12e} -> {after:.12e} (change {abs(after - before) / abs(before):.2e} of it); summed "
          f"size change {abs(float(state.size[0, joints].sum()) - summed) / summed:.2e} of it; pressed {int(pressed.sum())}")
    assert not pressed.any()
    assert abs(after - before) <= 1e-9 * abs(before)
    assert abs(float(state.size[0, joints].sum()) - summed) <= 1e-9 * summed


@needs_gpu
def test_later_substeps_over_listed_boxes_match_launches_over_all_boxes_to_the_last_bit(monkeypatch):
    module = sys.modules["physics.engine.substeps"]
    settings = Settings()
    runs = []
    for share in (module.DENSE_SHARE, -1.0):
        monkeypatch.setattr(module, "DENSE_SHARE", share)
        state = new_universe(4, 10, seed=3)
        quanta = state.quanta.numpy()
        place = tuple(np.argwhere(quanta > 0)[quanta[quanta > 0].size // 2])
        quanta[place] *= 30
        state.quanta.assign(quanta)
        listed, dense = 0, 0
        for _ in range(5):
            step(state, settings)
            listed += 0 if state.later["boxes"] is None else state.later["boxes"].shape[0]
            dense = max(dense, state.later["dense"])
        runs.append((state.quanta.numpy(), state.phase.numpy(), state.owner.numpy(), listed, dense))
    print(f"listed boxes over 5 steps {runs[0][3]} (launches over all boxes a step: {runs[0][4]}); all boxes: "
          f"{runs[1][4]} launches a step")
    assert runs[0][3] > 0 and runs[0][4] == 1
    assert runs[1][3] == 0 and runs[1][4] > 1
    for first, second in zip(runs[0][:3], runs[1][:3]):
        assert np.array_equal(first, second)
