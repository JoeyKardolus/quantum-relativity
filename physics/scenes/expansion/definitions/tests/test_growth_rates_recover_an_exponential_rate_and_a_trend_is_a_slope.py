"""A size growing as e^(0.3 t) has rate 0.3 everywhere; a rate rising as 2 t has trend 2."""
import math

import numpy as np
import pytest

from physics.scenes.expansion.definitions.growth_rates import growth_rates
from physics.scenes.expansion.definitions.rate_trend import rate_trend


def test_exponential_growth_reads_its_rate_at_every_midpoint():
    times = np.array([0.0, 1.0, 3.0, 4.0])
    midpoints, rates = growth_rates(times, 7.0 * np.exp(0.3 * times))
    assert midpoints.tolist() == [0.5, 2.0, 3.5]
    assert rates == pytest.approx([0.3, 0.3, 0.3])


def test_a_shrinking_size_has_a_negative_rate_and_bad_samples_raise():
    assert growth_rates([0.0, 1.0], [2.0, 1.0])[1][0] == pytest.approx(-math.log(2.0))
    for times, sizes in (([0.0], [1.0]), ([0.0, 0.0], [1.0, 2.0]), ([0.0, 1.0], [1.0, 0.0])):
        with pytest.raises(ValueError):
            growth_rates(times, sizes)


def test_a_trend_is_the_slope_of_rate_against_time():
    assert rate_trend([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]) == pytest.approx(2.0)
    assert rate_trend([0.0, 1.0, 2.0], [5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(rate_trend([1.0], [1.0]))
