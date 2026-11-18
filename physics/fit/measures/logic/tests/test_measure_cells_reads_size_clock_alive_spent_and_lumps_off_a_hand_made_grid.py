"""On a hand-made grid whose every number is known, measure_cells gives the size, the observer's clock, what is alive and spent, and the lumps."""
import numpy as np
import pytest

from physics.engine import NO_CELL
from physics.fit.measures.logic.measure_cells import measure_cells
from physics.fit.measures.logic.tests.hand_cells import lattice_cells


def test_an_even_grid_is_all_alive_nothing_spent_and_has_no_lump():
    measured = measure_cells(lattice_cells(), 20.0)
    assert measured["volume_size"] == 80.0 and measured["joint_size"] == 56 * 30.0 and measured["size"] == 1760.0
    assert (measured["plus"], measured["minus"], measured["amount"]) == (800, 560, 1360)
    assert (measured["volumes"], measured["joints"], measured["endless"]) == (8, 56, 0)
    assert measured["alive"] == 1.0 and measured["spent_share"] == 0.0 and measured["observer_rate"] == 1.0
    assert (measured["lump_volumes"], measured["lumps"], measured["largest_lump"]) == (0, 0, 0)
    assert measured["lump_share"] == 0.0 and measured["lump_spread"] == 0.0


def test_the_observers_clock_is_where_the_quanta_sit():
    cells = lattice_cells()
    cells["held"][0], cells["rate"][0] = 300, 0.5
    assert measure_cells(cells, 20.0)["observer_rate"] == pytest.approx((300 * 0.5 + 700 * 1.0) / 1000)


def test_lump_volumes_joined_by_a_way_are_one_lump_and_their_share_is_of_the_plus_quanta():
    cells = lattice_cells()
    cells["held"][[0, 2, 40]] = [300, 250, 200]
    measured = measure_cells(cells, 20.0)
    assert (measured["lump_volumes"], measured["lumps"], measured["largest_lump"]) == (3, 2, 2), "0 and 2 share a way"
    assert measured["lump_share"] == pytest.approx(750 / 1250)
    assert measured["lump_spread"] == pytest.approx(float(np.std(np.log([300, 250, 200]))))


def test_a_volume_that_fell_into_another_is_not_alive():
    cells = lattice_cells()
    cells["kind"][8], cells["held"][8], cells["length"][8] = NO_CELL, 0, 0.0
    measured = measure_cells(cells, 20.0)
    assert measured["alive"] == 7 / 8 and measured["volumes"] == 7 and measured["volume_size"] == 70.0


def test_spent_is_the_share_of_the_volumes_size_in_volumes_past_the_tear_length():
    cells = lattice_cells()
    cells["length"][10] = 30.0
    assert measure_cells(cells, 20.0)["spent_share"] == pytest.approx(30.0 / 100.0)
    assert measure_cells(cells, 30.0)["spent_share"] == 0.0, "a volume exactly at the tear length is not spent"
    assert measure_cells(cells, 5.0)["spent_share"] == 1.0, "the joints, 30 long, are never counted as spent"


def test_an_emptied_volume_is_endless_so_everything_is_spent_and_the_size_stays_a_number():
    cells = lattice_cells()
    cells["held"][10], cells["length"][10] = 0, np.inf
    measured = measure_cells(cells, 20.0)
    assert measured["spent_share"] == 1.0 and measured["endless"] == 1
    assert measured["size"] == 1750.0, "the endless cell is left out of the size and counted"


def test_a_universe_whose_volumes_hold_nothing_cannot_be_measured():
    cells = lattice_cells()
    cells["held"][cells["kind"] == 1] = 0
    with pytest.raises(ValueError, match="nothing to measure"):
        measure_cells(cells, 20.0)
