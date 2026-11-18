"""The hold factor is alive x (1 - spent share): a straight line in each, no cliff, clipped to shares."""
import pytest

from physics.fit.score.definitions.hold_factor import hold_factor


def test_a_whole_calm_universe_holds_fully_and_each_loss_costs_its_own_share():
    assert hold_factor(1.0, 0.0) == 1.0
    assert [hold_factor(alive, 0.0) for alive in (1.0, 0.75, 0.5, 0.25, 0.0)] == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert [hold_factor(1.0, spent) for spent in (0.0, 0.25, 0.5, 0.75, 1.0)] == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert hold_factor(0.5, 0.5) == 0.25


def test_one_volume_left_of_1728_holds_about_nothing_and_one_black_hole_costs_one_volume():
    assert hold_factor(1.0 / 1728.0, 0.0) == pytest.approx(0.000579, abs=1e-6)
    assert hold_factor(1727.0 / 1728.0, 0.0) == pytest.approx(0.999421, abs=1e-6)


def test_values_outside_a_share_are_clipped():
    assert hold_factor(1.5, -0.2) == 1.0 and hold_factor(-0.1, 0.0) == 0.0 and hold_factor(1.0, 1.7) == 0.0
