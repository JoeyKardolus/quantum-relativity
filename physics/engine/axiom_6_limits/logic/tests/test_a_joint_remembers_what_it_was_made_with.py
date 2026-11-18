"""start_fresh: every joint's fresh quanta are what it holds at the start, and nothing else is written."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_6_limits.logic.start_fresh import start_fresh
from physics.engine.axiom_6_limits.logic.tests.conftest import JOINT_BOX


def test_every_joint_is_fresh_at_what_it_holds_and_a_volume_has_no_fresh_quanta(state):
    kind, fresh = state.host("kind"), state.host("fresh_quanta")
    assert (fresh[kind == JOINT] == JOINT_BOX * 216).all()
    assert (fresh[kind == ENERGY_VOLUME] == 0).all()


def test_fresh_follows_each_joint_s_own_quanta_and_writes_nothing_else(state):
    state.set_cell("cell_quanta", 16, 777)
    state.set_cell("fresh_quanta", 0, 5)                 # a wrong value on a volume is wiped
    before = {name: getattr(state, name).numpy().copy() for name in ("quanta", "phase", "cell_quanta", "kind")}
    start_fresh(state)
    assert state.host("fresh_quanta")[16] == 777 and state.host("fresh_quanta")[0] == 0
    for name, values in before.items():
        assert np.array_equal(getattr(state, name).numpy(), values), name
