"""A unit-cube point maps 0 and 1 to the ends of each range, and the middle to the geometric mean of a log range and the plain mean of a linear one."""
import math

import pytest

from physics.fit.search.definitions.ranges import LINEAR_RANGES, LOG_RANGES, SEARCHED
from physics.fit.search.definitions.unit_to_overrides import unit_to_overrides


def test_the_corners_are_the_ends_of_the_ranges():
    lows, highs = unit_to_overrides([0.0] * 10), unit_to_overrides([1.0] * 10)
    for name, (low, high) in (LOG_RANGES | LINEAR_RANGES).items():
        assert lows[name] == pytest.approx(low) and highs[name] == pytest.approx(high)
    assert list(lows) == SEARCHED


def test_the_middle_is_the_geometric_mean_of_a_log_range_and_the_mean_of_a_linear_one():
    middle = unit_to_overrides([0.5] * 10)
    assert middle["pull_strength"] == pytest.approx(math.sqrt(1e-9 * 1e-4))
    assert middle["friction"] == pytest.approx(0.1) and middle["volume_quanta"] == pytest.approx(math.sqrt(1e7))
    assert middle["unevenness"] == pytest.approx(0.45)


def test_a_point_with_the_wrong_number_of_coordinates_is_refused():
    with pytest.raises(ValueError, match="10 coordinates"):
        unit_to_overrides([0.5] * 9)
