"""deceleration recovers q = (1 - p) / p for a size growing as t**p, and 0 for a straight line."""
import math

import numpy as np
import pytest

from physics.scenes import deceleration


def test_quadratic_growth_speeds_up_with_q_minus_one_half():
    times = np.linspace(1.0, 2.0, 11)
    assert deceleration(times, times ** 2) == pytest.approx(-0.5, abs=1e-9)


def test_square_root_growth_slows_with_q_near_one():
    times = np.linspace(1.0, 1.2, 11)
    assert deceleration(times, np.sqrt(times)) == pytest.approx(1.0, abs=0.02)


def test_straight_growth_has_q_zero_and_too_few_samples_give_nan():
    times = np.linspace(0.0, 5.0, 6)
    assert deceleration(times, 3.0 + 2.0 * times) == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(deceleration([0.0, 1.0], [1.0, 2.0]))


def test_times_that_do_not_rise_raise():
    with pytest.raises(ValueError):
        deceleration([0.0, 0.0, 1.0], [1.0, 2.0, 3.0])
