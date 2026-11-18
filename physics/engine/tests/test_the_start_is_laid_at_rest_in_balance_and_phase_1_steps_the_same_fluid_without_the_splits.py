"""The start rule (at rest, in balance) on real universes, the public door for a placed start, and the phase 1 step."""
from __future__ import annotations

import numpy as np
import pytest

from physics.engine import ENERGY_VOLUME, JOINT, Settings, lay_start, new_universe, step
from physics.engine.tests.conftest import needs_gpu

pytestmark = needs_gpu


def _universe(**changed):
    settings = Settings(**changed)
    return new_universe(settings, cells_per_side=8, seed=1, device="cuda"), settings


def test_the_default_universe_starts_with_every_joint_in_balance_and_the_pull_at_rest():
    state, settings = _universe()
    kind, squeeze, length = state.host("kind"), state.host("squeeze"), state.host("length")
    assert 0.995 < squeeze[kind == JOINT].min() and squeeze[kind == JOINT].max() <= 1.0
    assert state.pull_unrest < 1e-6 * np.ptp(state.host("pull"))
    ratio = length[kind == JOINT].mean() / length[kind == ENERGY_VOLUME].mean()
    print(f"default start: a joint is {ratio:.2f} times as long as a volume")
    assert 1.0 < ratio < 4.0, "the balance, not a setting, says how thick a joint is (measured 2.2)"


def test_the_default_universe_stands_no_collapse_and_no_tear_in_its_first_hundred_steps():
    state, settings = _universe()
    for _ in range(100):
        step(state, settings, quantum=False)
    assert state.collapses == 0 and state.tears == 0
    assert int((state.host("kind") == ENERGY_VOLUME).sum()) == 64


def test_a_start_with_energy_placed_by_hand_is_laid_again_and_only_a_start_is():
    state, settings = _universe(unevenness=0.0)
    volume = int(np.flatnonzero(state.host("kind") == ENERGY_VOLUME)[0])
    state.set_block("quanta", volume, state.block("quanta", volume) * 5)
    lay_start(state, settings)
    kind, squeeze = state.host("kind"), state.host("squeeze")
    assert squeeze[kind == JOINT].max() <= 1.0 and squeeze[kind == JOINT].min() > 0.995
    assert state.host("pull")[volume] == state.host("pull")[kind == ENERGY_VOLUME].min(), "the fuller volume sits deepest"
    step(state, settings)
    with pytest.raises(ValueError, match="lays a start"):
        lay_start(state, settings)


def test_phase_1_steps_the_same_fluid_and_opens_no_split():
    whole, settings = _universe()
    below, _ = _universe()
    for _ in range(6):
        step(whole, settings)
        step(below, settings, quantum=False)
    assert np.array_equal(whole.quanta.numpy(), below.quanta.numpy()), "no split settled, so the fluid is the same"
    assert np.array_equal(whole.phase.numpy(), below.phase.numpy())
    assert len(whole.quantum["source"]) > 0 and len(below.quantum["source"]) == 0
