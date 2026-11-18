"""In a huge box the law's tilt quarters when the distance doubles; in a small one the copies weaken it; an extra centre is a weighted mean."""
import pytest

from physics.scenes.two_lumps.definitions.extra_centre import extra_centre
from physics.scenes.two_lumps.definitions.newton_tilt import newton_tilt


def test_the_tilt_is_one_over_distance_squared_when_the_copies_are_far():
    near, far = newton_tilt(10.0, 0.5, 1e6), newton_tilt(20.0, 0.5, 1e6)
    assert near == pytest.approx(1.0 / 100.0, rel=0.01)
    assert near / far == pytest.approx(4.0, rel=0.01)


def test_the_copies_weaken_the_tilt_toward_half_the_box():
    assert 0.0 < newton_tilt(40.0, 1.0, 100.0) < 0.9 / 40.0 ** 2
    assert newton_tilt(50.0, 1.0, 100.0) == pytest.approx(0.0, abs=2e-6), "half way to the copy the two pulls cancel"
    with pytest.raises(ValueError):
        newton_tilt(1.0, 2.0, 100.0)


def test_an_extra_centre_is_the_energy_weighted_mean_place():
    assert extra_centre([0.0, 0.0, 5.0], [-1, 0, 1]) == 1.0
    assert extra_centre([1.0, 2.0, 1.0], [-1, 0, 1]) == 0.0
    assert extra_centre([3.0, 0.0, 1.0], [-1, 0, 1]) == -0.5
    assert extra_centre([0.0, 0.0, 0.0], [-1, 0, 1]) == 0.0
