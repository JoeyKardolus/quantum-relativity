"""`balance_level` on a small lattice with the press set by hand, and on a real start."""
from __future__ import annotations

import numpy as np
import pytest

from physics import engine
from physics.engine.axiom_5_packing.logic.balance_level import balance_level
from physics.engine.axiom_5_packing.logic.tests.conftest import JOINT, fill_even, set_cells


def test_the_level_is_the_fourth_root_of_the_press_over_the_push_s_strength(state, settings):
    fill_even(state, settings)
    kind = state.host("kind")
    press = np.where(kind == JOINT, 5.0, 9.0).astype(np.float32)
    joints = np.flatnonzero(kind == JOINT)
    press[joints[0]], press[joints[1]] = 0.0, -3.0
    set_cells(state, "press", press)
    level = balance_level(state, settings)
    wanted = (5.0 * settings.size_constant ** 2 / settings.push_strength) ** 0.25
    assert level[joints[2]] == pytest.approx(wanted) and level[joints[0]] == 0.0 and level[joints[1]] == 0.0
    assert not level[kind != JOINT].any(), "only a joint has a level"
    push = settings.push_strength * (wanted / (settings.size_constant / wanted)) ** 2
    assert push == pytest.approx(5.0), "at the level the joint's own push is the press"


def test_a_real_start_stands_at_its_level_no_joint_squeezed_and_none_slack_by_a_thousandth():
    settings = engine.Settings()
    state = engine.new_universe(settings, cells_per_side=8, seed=1, device="cpu")
    joints = state.host("kind") == JOINT
    held, level = state.host("cell_quanta")[joints].astype(np.float64), balance_level(state, settings)[joints]
    assert (level <= held).all() and (level > 0.999 * held).all()
    assert state.balance_off <= 1e-4 and state.host("squeeze")[joints].max() <= 1.0
