"""Axiom 2's step check on a real small universe: 0 at a real start, and each wrong made by hand is counted."""
from __future__ import annotations

import numpy as np
import warp as wp

from physics import engine
from physics.engine.axiom_2_swirls.tests.check_step import check_step

wp.config.quiet = True

WRONGS = ("against_volumes", "against_joints", "unwound_volumes", "unwound_joints", "joint_pull_made",
          "joint_pull_felt")


def _started():
    settings = engine.Settings()
    return engine.new_universe(settings, cells_per_side=4, seed=1, device="cpu"), settings


def test_a_real_start_has_every_cell_going_round_its_own_way():
    state, settings = _started()
    counts = check_step(state, settings)
    assert [counts[name] for name in WRONGS] == [0] * len(WRONGS)
    assert counts["round_volumes"] > 0.1 and counts["round_joints"] > 0.1


def test_a_cell_whose_fluid_is_turned_round_by_hand_is_counted_as_going_against_its_sense():
    state, settings = _started()
    kind = state.host("kind")
    joint, volume = int((kind == engine.JOINT).argmax()), int((kind == engine.ENERGY_VOLUME).argmax())
    for cell in (joint, volume):
        state.set_block("phase", cell, -state.block("phase", cell))
    counts = check_step(state, settings)
    assert counts["against_volumes"] == 1 and counts["against_joints"] == 1
    assert counts["unwound_volumes"] == 1 and counts["unwound_joints"] == 1


def test_a_cell_whose_fluid_has_one_phase_is_no_swirl_and_a_pull_ledger_on_a_joint_is_counted():
    state, settings = _started()
    kind = state.host("kind")
    joint, volume = int((kind == engine.JOINT).argmax()), int((kind == engine.ENERGY_VOLUME).argmax())
    state.set_block("phase", volume, np.zeros((engine.BLOCK,) * 3))
    state.set_cell("pull_made", joint, 1e-3)
    state.set_cell("pull_felt", joint, 1e-3)
    counts = check_step(state, settings)
    assert counts["against_volumes"] == 1 and counts["unwound_volumes"] == 1 and counts["against_joints"] == 0
    assert counts["joint_pull_made"] == 1 and counts["joint_pull_felt"] == 1
