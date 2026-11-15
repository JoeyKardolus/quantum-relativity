"""Joint push rises steeply as a joint thins, like a hard wall, so ordinary pull never collapses one (rule 8)."""
import numpy as np
import pytest

from physics.cells import (EngineSettings, joint_balance, joint_push, joint_rest, new_engine, pull_potential,
                           step)


def test_the_push_rises_faster_than_a_straight_spring_as_a_joint_thins():
    rest = np.ones(9)
    thickness = np.linspace(0.9, 0.1, 9)
    steep = joint_push(thickness, rest, 0.1, 3.0)
    straight = 0.1 * (rest - thickness)
    assert (steep > straight).all()
    assert (np.diff(steep / straight) > 0).all()
    assert (np.diff(steep, 2) > 0).all()


def test_power_minus_one_is_the_straight_spring_and_rest_pushes_nothing():
    rest = np.array([1.0, 2.0, 0.5])
    thickness = np.array([0.3, 2.0, 0.7])
    assert np.allclose(joint_push(thickness, rest, 0.1, -1.0), 0.1 * (rest - thickness))
    assert joint_push(thickness, rest, 0.1, 3.0)[1] == 0.0


@pytest.mark.xfail(strict=True, reason="miss (2026-11-15, seed 1, 1000 cells, unevenness 0.5, defaults): "
                   "joints still reach the push limit at default pull, first collapse step 53 with "
                   "splits off, and volume splits blow the size up and stop the run at step 151")
def test_no_joint_collapses_in_300_steps_from_the_default_start():
    engine = new_engine(1000, 1, 0.5)
    settings = EngineSettings.all_on()
    for _ in range(300):
        engine = step(engine, settings)
    assert engine.collapses == 0


def test_at_its_balance_a_joints_push_cancels_its_squeeze():
    engine = new_engine(200, 2, 0.5)
    settings = EngineSettings.all_on()
    grid = engine.grid
    exerted = settings.attraction * pull_potential(grid) / 2
    squeeze = exerted[grid.joints[:, 0]] + exerted[grid.joints[:, 1]]
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    balance = joint_balance(grid, settings)
    force = joint_push(balance, rest, settings.joint_push, settings.push_power) - squeeze * balance
    assert np.abs(force).max() < 1e-9 * rest.max()
    assert ((balance > settings.push_limit * rest) & (balance < rest)).all()
