"""The second half change is last over middle; the mark wants a lump within 10% of its length while the rest grows over 5%."""
import pytest

from physics.scenes.bound_lump.definitions.pass_mark import passes
from physics.scenes.bound_lump.definitions.second_half_change import second_half_change

GOOD = {"is_lump": 1, "lump_length": [5.0, 4.0, 4.0, 4.1, 4.2], "rest_length": [10.0, 10.0, 10.0, 10.5, 11.0]}


def test_the_change_is_read_from_the_middle_sample_on():
    assert second_half_change([5.0, 4.0, 4.0, 4.1, 4.2]) == pytest.approx(1.05)
    assert second_half_change([3.0]) == 1.0
    with pytest.raises(ValueError):
        second_half_change([])


def test_the_mark_passes_a_kept_lump_in_growing_space_and_misses_the_rest():
    assert passes(GOOD, True, True)
    assert not passes({**GOOD, "is_lump": 0}, True, True), "the fullest volume is no lump"
    assert not passes({**GOOD, "lump_length": [5.0, 4.0, 4.0, 4.4, 4.8]}, True, True), "the lump grew 20%"
    assert not passes({**GOOD, "rest_length": [10.0, 10.0, 10.0, 10.1, 10.2]}, True, True), "space did not grow"
    assert not passes(GOOD, False, True) and not passes(GOOD, True, False)
