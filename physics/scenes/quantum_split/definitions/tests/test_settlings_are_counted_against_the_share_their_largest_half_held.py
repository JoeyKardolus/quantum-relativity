"""A 70/30 split's largest half holds 0.7; settlings in that cell are counted beside the mean share; the mark wants the two within 0.05."""
import numpy as np
import pytest

from physics.scenes.quantum_split.definitions.landing import landing
from physics.scenes.quantum_split.definitions.largest_half import largest_half
from physics.scenes.quantum_split.definitions.pass_mark import NOT_ASKED, passes

SOURCES = [10, 20, 30]
HALVES = ([10, 10, 20, 20, 20, 30, 30], [10, 12, 20, 22, 24, 30, 32], [700, 300, 500, 250, 250, 400, 400])


def test_each_splits_largest_half_and_its_share():
    cells, shares = largest_half(SOURCES, *HALVES)
    assert cells.tolist() == [10, 20, 30], "a tie goes to the lower cell id"
    assert shares.tolist() == [0.7, 0.5, 0.5]
    empty_cells, empty_shares = largest_half([5], [], [], [])
    assert empty_cells.tolist() == [-1] and empty_shares.tolist() == [0.0]
    with pytest.raises(ValueError):
        largest_half([10], [11], [11], [5])


def test_landing_counts_only_settlings_whose_split_was_read():
    cells, shares = largest_half(SOURCES, *HALVES)
    found = landing([10, 20, 30, 40], [10, 22, 30, 40], SOURCES, cells, shares)
    assert found["judged"] == 3 and found["unread"] == 1
    assert found["found"] == pytest.approx(2.0 / 3.0) and found["expected"] == pytest.approx((0.7 + 0.5 + 0.5) / 3.0)
    assert landing([], [], SOURCES, cells, shares) == {"judged": 0, "unread": 0, "found": 0.0, "expected": 0.0}


def test_a_thousand_draws_by_share_land_in_the_largest_half_as_often_as_its_share():
    draws = np.random.default_rng(7).random(1000)
    sources = np.arange(1000)
    cells, shares = np.zeros(1000, dtype=np.int64), np.full(1000, 0.7)
    found = landing(sources, np.where(draws < 0.7, 0, 1), sources, cells, shares)
    assert found["found"] == pytest.approx(0.7, abs=0.04) and found["expected"] == pytest.approx(0.7)
    result = {"judged": 1000, "whole_share": 1.0, "found_in_largest": found["found"], "expected_in_largest": 0.7}
    assert passes(result)
    assert not passes({**result, "found_in_largest": 1.0}), "always the largest half is not by share"
    assert not passes({**result, "judged": 150}) and not passes({**result, "whole_share": 0.99})
    assert len(NOT_ASKED) == 3
