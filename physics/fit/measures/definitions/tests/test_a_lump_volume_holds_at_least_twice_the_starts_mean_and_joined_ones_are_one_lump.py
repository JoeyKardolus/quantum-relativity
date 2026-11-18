"""A lump volume is an energy volume holding at least twice the start's mean, and lump volumes joined through ways are one lump."""
import numpy as np

from physics.engine import ENERGY_VOLUME, JOINT, NO_CELL
from physics.fit.measures.definitions.lump_groups import lump_groups
from physics.fit.measures.definitions.lump_marks import LUMP_FACTOR
from physics.fit.measures.definitions.lump_volumes import lump_volumes


def test_the_mark_is_twice_the_starts_mean_and_only_an_energy_volume_can_pass_it():
    kind = np.array([ENERGY_VOLUME, ENERGY_VOLUME, ENERGY_VOLUME, JOINT, NO_CELL])
    held = np.array([199, 200, 900, 900, 900])
    assert LUMP_FACTOR == 2
    assert lump_volumes(kind, held, 100.0).tolist() == [False, True, True, False, False]


def test_lump_volumes_with_a_way_between_them_are_one_lump_and_the_rest_stand_alone():
    packed = np.array([True, True, False, True, False, True])
    first, second = np.arange(6), (np.arange(6) + 1) % 6
    numbered, lumps = lump_groups(6, packed, first, second)
    assert lumps == 2, "0, 1 and 5 hang together round the ring; 3 stands alone"
    assert numbered[0] == numbered[1] == numbered[5] != numbered[3]


def test_a_chain_of_lump_volumes_is_one_lump_and_no_lump_volume_gives_none():
    first, second = np.array([0, 1, 2]), np.array([1, 2, 3])
    assert lump_groups(4, np.ones(4, dtype=bool), first, second)[1] == 1
    numbered, lumps = lump_groups(4, np.zeros(4, dtype=bool), first, second)
    assert lumps == 0 and not numbered.any()
