"""The gap between lumps is the summed lengths of the same cells between them, chosen once by fewest cells, never by length."""
import pytest

from physics.fit.measures.logic.gap_between_lumps import gap_between_lumps
from physics.fit.measures.logic.tests.hand_cells import ring_snapshot

_EVEN = [10.0] * 6 + [1.0] * 6


def test_the_gap_sums_the_same_cells_between_two_lumps_in_every_snapshot_and_not_the_lumps_themselves():
    grown = [10.0, 30.0, 10.0, 10.0, 10.0, 10.0] + [1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    last = ring_snapshot(grown, lumps=[0, 2], held=[300, 20, 300, 60, 60, 60] + [10] * 6)
    gaps = gap_between_lumps([ring_snapshot(_EVEN, lumps=[]), last])
    assert gaps[0] == pytest.approx(2 * 1.0 + 10.0), "two joints and one volume lie between, the way with the fewest cells"
    assert gaps[1] == pytest.approx(1.0 + 2.0 + 30.0), "the same three cells, though the other way round is now shorter"


def test_the_way_between_is_chosen_by_cells_not_by_length_so_evening_out_does_not_lengthen_it():
    uneven = [10.0, 40.0, 10.0, 5.0, 5.0, 5.0] + [1.0] * 6
    evened = [10.0, 20.0, 10.0, 12.0, 12.0, 12.0] + [1.0] * 6
    last = ring_snapshot(evened, lumps=[0, 2], held=[300, 20, 300, 60, 60, 60] + [10] * 6)
    gaps = gap_between_lumps([ring_snapshot(uneven, lumps=[]), last])
    assert gaps == [pytest.approx(42.0), pytest.approx(22.0)], "through volume 1 both times, not the short way of the start"


def test_with_fewer_than_two_lumps_at_the_end_there_is_no_gap():
    assert gap_between_lumps([ring_snapshot(_EVEN, lumps=[]), ring_snapshot(_EVEN, lumps=[2])]) == [None, None]
    assert gap_between_lumps([ring_snapshot(_EVEN, lumps=[0, 1]), ring_snapshot(_EVEN, lumps=[0, 1])]) == [None, None], \
        "two lump volumes joined by a way are one lump"


def test_two_lump_volumes_joined_through_a_way_are_one_lump_with_one_centre():
    last = ring_snapshot(_EVEN, lumps=[0, 1, 3], held=[200, 250, 20, 300, 15, 15] + [10] * 6)
    gaps = gap_between_lumps([last, last])
    assert gaps[1] == pytest.approx(2 * 1.0 + 10.0), "from volume 1, the fuller of the joined pair, to volume 3"


def test_three_lumps_give_the_mean_over_the_three_pairs():
    last = ring_snapshot(_EVEN, lumps=[0, 2, 4], held=[300, 20, 300, 20, 300, 20] + [10] * 6)
    assert gap_between_lumps([last])[0] == pytest.approx(12.0), "each pair has one volume and two joints between"
