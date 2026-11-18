"""A length's shape is grew x steady x (1 + dip) / 2: straight growth earns half, slowing then speeding up earns all, shrinking earns nothing."""
import math

import numpy as np
import pytest

from physics.fit.score.definitions.length_shape import length_shape

_TAU = np.arange(7.0)


def test_growth_in_a_straight_line_is_steady_with_no_dip_and_earns_half():
    shape = length_shape(_TAU, [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0])
    assert shape["growth"] == pytest.approx(0.3) and shape["dip"] == pytest.approx(0.0, abs=1e-12)
    assert shape["shape"] == pytest.approx(math.tanh(3.0) / 2.0)


def test_growth_that_slows_and_then_speeds_up_has_a_known_dip_and_earns_the_full_shape():
    shape = length_shape(_TAU, [100.0, 110.0, 115.0, 116.0, 117.0, 122.0, 132.0])
    assert shape["dip"] == pytest.approx(13.0 / (32.0 / 3.0)), "thirds gain 15, 2, 15"
    assert shape["shape"] == pytest.approx(math.tanh(3.2)), "the dip is clipped to 1"


def test_growth_that_only_speeds_up_has_a_negative_dip_and_earns_less_than_a_straight_line():
    shape = length_shape(_TAU, [100.0, 101.0, 102.0, 104.0, 106.0, 118.0, 130.0])
    assert shape["dip"] == pytest.approx((2.0 - 4.0) / 10.0), "thirds gain 2, 4, 24"
    assert shape["shape"] == pytest.approx(math.tanh(3.0) * (1.0 - 0.2) / 2.0)


def test_growth_that_turns_back_in_between_is_less_steady_by_the_way_it_went():
    shape = length_shape(_TAU, [100.0, 120.0, 110.0, 130.0, 115.0, 135.0, 130.0])
    moved = 20 + 10 + 20 + 15 + 20 + 5
    thirds = np.array([10.0, 5.0, 15.0])
    dip = min(thirds[0] - thirds[1], thirds[2] - thirds[1]) / thirds.mean()
    assert shape["shape"] == pytest.approx(math.tanh(3.0) * (30 / moved) * (1.0 + dip) / 2.0)


def test_a_length_that_shrinks_or_stands_still_earns_nothing_and_one_starting_at_nothing_is_refused():
    assert length_shape(_TAU, [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0])["shape"] == 0.0
    assert length_shape(_TAU, [100.0] * 7) == {"shape": 0.0, "growth": 0.0, "dip": 0.0}
    with pytest.raises(ValueError, match="start above 0"):
        length_shape(_TAU, [0.0] * 7)


def test_the_thirds_are_cut_on_the_observers_time_not_on_the_samples():
    lengths = [100.0, 110.0, 120.0, 130.0]
    even = length_shape(np.array([0.0, 1.0, 2.0, 3.0]), lengths)
    slowed = length_shape(np.array([0.0, 1.0, 2.0, 6.0]), lengths)
    assert even["dip"] == pytest.approx(0.0, abs=1e-12)
    assert slowed["dip"] == pytest.approx((5.0 - 5.0) / 10.0), "thirds end at tau 2, 4, 6: lengths 120, 125, 130"
    slow_start = length_shape(np.array([0.0, 4.0, 5.0, 6.0]), lengths)
    assert slow_start["dip"] == pytest.approx((5.0 - 5.0) / 10.0) and slow_start["shape"] == pytest.approx(math.tanh(3.0) / 2)
