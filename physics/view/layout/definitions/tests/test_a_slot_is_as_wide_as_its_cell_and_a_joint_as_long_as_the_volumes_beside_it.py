"""What a slot, a width and a drawn length are, on numbers worked out by hand."""
import numpy as np
import pytest

from physics.view.layout.definitions.drawn_length import drawn_length
from physics.view.layout.definitions.fit_marks import FLOOR, MOST_TURNS, RELATIVE, SHEAR
from physics.view.layout.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL
from physics.view.layout.definitions.lattice_place import lattice_place
from physics.view.layout.definitions.pieces import Pieces
from physics.view.layout.definitions.slot_widths import slot_widths


def test_a_slot_number_is_its_place_with_z_running_fastest():
    assert lattice_place(np.array([0, 1, 4, 16, 63]), 4).tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [3, 3, 3]]
    assert lattice_place(np.array([8 * 8 * 5 + 8 * 2 + 7]), 8).tolist() == [[5, 2, 7]]


def test_a_volume_is_its_length_every_way_and_a_joint_only_along_its_odd_axes():
    side = 4
    odd = (np.indices((side, side, side)).reshape(3, -1) % 2).sum(axis=0)
    length = np.where(odd == 0, 10.0, 1.0)
    widths = slot_widths(side, length, length, np.zeros(side ** 3, dtype=int))
    assert widths[:, 0, 0, 0].tolist() == [10.0, 10.0, 10.0], "an energy volume"
    assert widths[:, 1, 0, 0].tolist() == [1.0, 10.0, 10.0], "a face joint is thin one way"
    assert widths[:, 1, 1, 0].tolist() == [1.0, 1.0, 10.0], "an edge joint two ways"
    assert widths[:, 1, 1, 1].tolist() == [1.0, 1.0, 1.0], "a corner joint all three"
    assert widths[0].sum(axis=0).min() == widths[0].sum(axis=0).max() == 22.0, "every row is as long as the grid"


def test_a_joint_between_a_big_and_a_small_volume_is_as_long_as_their_mean():
    side = 4
    odd = (np.indices((side, side, side)).reshape(3, -1) % 2).sum(axis=0)
    length = np.where(odd == 0, 10.0, 1.0)
    length[0] = 4.0
    widths = slot_widths(side, length, length, np.zeros(side ** 3, dtype=int))
    assert widths[1, 1, 0, 0] == 7.0 and widths[0, 1, 0, 0] == 1.0
    assert widths[1, 3, 0, 0] == 7.0, "the grid wraps round: the joint on the other side too"


def test_a_slots_stripe_axis_takes_the_summed_length_and_the_others_the_length_across():
    side = 4
    along, across = np.full(side ** 3, 10.0), np.full(side ** 3, 10.0)
    along[0], across[0] = 62.0, 30.0
    slot_axis = np.zeros(side ** 3, dtype=int)
    slot_axis[0] = 1
    assert slot_widths(side, along, across, slot_axis)[:, 0, 0, 0].tolist() == [30.0, 62.0, 30.0]


def test_a_gone_cell_is_drawn_with_no_length_and_an_endless_one_as_the_longest():
    length = np.array([10.0, np.inf, 0.0, 300.0, np.inf])
    alive = np.array([True, True, False, True, False])
    assert drawn_length(length, alive).tolist() == [10.0, 300.0, 0.0, 300.0, 0.0]
    assert drawn_length(np.array([np.inf]), np.array([True])).tolist() == [1.0], "nothing to measure against"


def test_the_kinds_the_marks_and_the_record_are_what_the_drawing_relies_on():
    assert (ENERGY_VOLUME, JOINT, NO_CELL) == (1, -1, 0)
    assert 0 < RELATIVE < 1e-4 and 0 < FLOOR < 0.01 and 0 < SHEAR < 1 and MOST_TURNS >= 100
    pieces = Pieces(np.zeros((0, 8, 3)), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 3)), 4, np.zeros(3), np.ones(3))
    with pytest.raises(AttributeError):
        pieces.side = 8
