"""The observer's time between two samples is the counter time times the mean of the two observer rates."""
import pytest

from physics.fit.measures.definitions.observer_time import observer_time


def test_a_clock_at_full_rate_keeps_counter_time_and_a_slowing_one_keeps_the_mean():
    assert observer_time(25.0, 1.0, 1.0) == 25.0
    assert observer_time(25.0, 1.0, 0.5) == pytest.approx(18.75)
    assert observer_time(25.0, 0.5, 1.0) == pytest.approx(18.75)
    assert observer_time(0.0, 0.3, 0.9) == 0.0
