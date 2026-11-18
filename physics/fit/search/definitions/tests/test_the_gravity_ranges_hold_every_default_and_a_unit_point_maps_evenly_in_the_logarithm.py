"""Phase 1's ranges: only gravity's settings, every default inside, the cube's ends at the ranges' ends."""
import math

import pytest

from physics.engine import Settings
from physics.fit.search.definitions.gravity_ranges import GRAVITY_RANGES, GRAVITY_SEARCHED, gravity_point


def test_every_searched_name_is_a_setting_whose_default_lies_inside_its_range():
    defaults = Settings()
    assert GRAVITY_SEARCHED == list(GRAVITY_RANGES) and len(GRAVITY_SEARCHED) == 7
    for name, (low, high) in GRAVITY_RANGES.items():
        assert 0.0 < low <= getattr(defaults, name) <= high, name
    assert not {"volume_quanta", "unevenness", "step_time", "size_constant"} & set(GRAVITY_SEARCHED)


def test_the_ends_of_the_cube_are_the_ends_of_the_ranges_and_the_middle_the_geometric_mean():
    low, high, middle = gravity_point([0.0] * 7), gravity_point([1.0] * 7), gravity_point([0.5] * 7)
    for name, (start, end) in GRAVITY_RANGES.items():
        assert low[name] == pytest.approx(start) and high[name] == pytest.approx(end)
        assert middle[name] == pytest.approx(math.sqrt(start * end))
    Settings(**middle)
    with pytest.raises(ValueError, match="7 coordinates"):
        gravity_point([0.5] * 6)
