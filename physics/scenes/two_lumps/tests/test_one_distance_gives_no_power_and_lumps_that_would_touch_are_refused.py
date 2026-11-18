"""A single distance reads a pull beside a law of exactly 1 and no power; distances that touch or wrap are refused."""
import math

import pytest

from physics.scenes.tests._gpu import HELD, needs_gpu
from physics.scenes.two_lumps import passes, two_lumps


@needs_gpu
def test_a_single_distance_is_read_beside_its_own_law_and_has_no_power():
    result = two_lumps(12, apart=(2,), steps=3, settle=1, settings=HELD)
    assert result["newton"] == [1.0] and result["apart"] == [2]
    assert result["pull"][0] == pytest.approx(result["tilt"][0] - result["tilt_alone"][0])
    assert math.isnan(result["pull_power"]) and math.isnan(result["newton_power"])
    assert not passes(result), "no distance of 3 volumes or more was read"


def test_lumps_one_volume_apart_or_over_a_third_of_the_box_apart_are_refused():
    for arguments in ({"apart": (1,)}, {"apart": (2, 5)}, {"apart": ()}, {"factor": 1.0}, {"steps": 0}):
        with pytest.raises(ValueError):
            two_lumps(**{"cells_per_side": 24, "apart": (2,), **arguments})
