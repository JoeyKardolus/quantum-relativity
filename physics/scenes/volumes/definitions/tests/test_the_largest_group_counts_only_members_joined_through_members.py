"""In a chain of five with the middle one no member, the largest group is two; a lump is two even volumes' worth."""
import numpy as np
import pytest

from physics.scenes.volumes.definitions.largest_group import largest_group
from physics.scenes.volumes.definitions.lump_factor import LUMP_FACTOR

CHAIN = (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]))


def test_a_chain_broken_by_a_non_member_gives_two_groups_of_two():
    assert largest_group(np.array([True, True, False, True, True]), *CHAIN) == 2
    assert largest_group(np.ones(5, dtype=bool), *CHAIN) == 5


def test_no_member_is_zero_and_unjoined_members_are_one_each():
    assert largest_group(np.zeros(5, dtype=bool), *CHAIN) == 0
    assert largest_group(np.array([True, False, True, False, True]), *CHAIN) == 1
    with pytest.raises(ValueError):
        largest_group(np.ones(3, dtype=bool), np.array([0]), np.array([1, 2]))


def test_the_lump_factor_is_two():
    assert LUMP_FACTOR == 2.0
