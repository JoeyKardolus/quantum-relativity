"""Joints oscillate: no damping, so they overshoot and keep swinging around their balance (rule 5)."""
import numpy as np
import pytest

from physics.cells import EngineSettings, joint_balance, new_engine, step, swing_joints


@pytest.mark.parametrize("power", [-1.0, 3.0])
def test_a_steady_joint_spring_keeps_its_swing_for_a_thousand_steps(power):
    thickness, rate = np.array([1.2]), np.array([0.0])
    rest, squeeze = np.array([1.5]), np.array([0.05])
    seen = []
    for _ in range(1000):
        thickness, rate = swing_joints(thickness, rate, rest, squeeze, 0.1, power)
        seen.append(thickness[0])
    late = np.ptp(seen[-100:])
    early = np.ptp(seen[:100])
    assert 0.9 < late / early < 1.1


def test_most_joints_cross_their_balance_more_than_once_in_50_steps():
    engine = new_engine(200, 5, 0.5)
    settings = EngineSettings.all_on(push_limit=0.0, split_threshold=0.0)
    signs = []
    for _ in range(50):
        engine = step(engine, settings)
        signs.append(np.sign(engine.grid.thickness - joint_balance(engine.grid, settings)))
    crossings = (np.diff(np.array(signs), axis=0) != 0).sum(axis=0)
    assert np.median(crossings) > 1
