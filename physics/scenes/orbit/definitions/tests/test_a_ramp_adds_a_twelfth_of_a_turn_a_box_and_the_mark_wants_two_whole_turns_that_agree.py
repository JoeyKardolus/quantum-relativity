"""The ramp rises pi / 6 a box along its axis and wraps; the mark wants a whole turn at two distances with Kepler ratios within 10%."""
import math

import numpy as np
import pytest

from physics.scenes.orbit.definitions.pass_mark import passes
from physics.scenes.orbit.definitions.sideways_ramp import SIDEWAYS, sideways_ramp


def test_the_ramp_rises_along_its_axis_only_and_stays_inside_one_turn():
    ramped = sideways_ramp(np.zeros((6, 6, 6)), SIDEWAYS, 1)
    assert SIDEWAYS == pytest.approx(math.pi / 6.0)
    assert ramped[0, :, 0] == pytest.approx([0.0, math.pi / 6, math.pi / 3, math.pi / 2, 2 * math.pi / 3, 5 * math.pi / 6])
    assert np.ptp(ramped, axis=0).max() == 0.0 and np.ptp(ramped, axis=2).max() == 0.0
    wrapped = sideways_ramp(np.full((6, 6, 6), 3.0), 1.0, 0)
    assert wrapped.max() <= math.pi and wrapped.min() > -math.pi
    assert np.exp(1j * wrapped[2, 0, 0]) == pytest.approx(np.exp(1j * 5.0))
    with pytest.raises(ValueError):
        sideways_ramp(np.zeros((6, 6, 6)), 1.0, 3)


def test_two_whole_turns_with_one_ratio_pass_and_no_whole_turn_misses():
    turn = {"turned": 1.2, "period_squared_over_radius_cubed": 50.0}
    assert passes({"runs": [turn, {"turned": -1.0, "period_squared_over_radius_cubed": 54.0}]})
    assert not passes({"runs": [turn, {"turned": 1.0, "period_squared_over_radius_cubed": 60.0}]}), "ratios 20% apart"
    assert not passes({"runs": [turn, {"turned": 0.9, "period_squared_over_radius_cubed": 50.0}]}), "one whole turn only"
    assert not passes({"runs": [{"turned": 0.011, "period_squared_over_radius_cubed": 0.0}] * 3})
