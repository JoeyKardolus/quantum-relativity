"""The gap between lumps is the summed lengths of the cells between them along the shortest chain of ways, and held lumps are those still lumps at the end."""
import numpy as np
import pytest

from physics.fit.fit_gap import fit_gap


def _ring(lengths, lumps, held=None):
    """Six volumes (ids 0..5) in a ring with a joint (ids 6..11) between each pair of neighbours."""
    volumes = np.arange(6)
    ways = (volumes + 6, volumes, (volumes + 1) % 6)
    held = np.array([100, 100, 100, 100, 100, 100] + [10] * 6 if held is None else held, dtype=np.int64)
    return {"held": held, "length": np.asarray(lengths, dtype=np.float32), "count": 12, "plus": int(held[:6].sum()),
            "lumps": np.asarray(lumps, dtype=np.int64), "ways": ways, "version": 0}


def test_the_gap_sums_the_same_cells_between_two_lumps_in_every_snapshot_and_not_the_lumps_themselves():
    even = [10.0] * 6 + [1.0] * 6
    grown = [10.0, 30.0, 10.0, 10.0, 10.0, 10.0] + [1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    last = _ring(grown, lumps=[0, 2], held=[300, 20, 300, 60, 60, 60] + [10] * 6)
    gaps, _ = fit_gap([_ring(even, lumps=[]), last], {})
    assert gaps[0] == pytest.approx(2 * 1.0 + 10.0), "two joints and one volume lie between, the way with the fewest cells"
    assert gaps[1] == pytest.approx(1.0 + 2.0 + 30.0), "the same three cells, though the other way round is now shorter"


def test_the_way_between_is_chosen_by_cells_not_by_length_so_evening_out_does_not_lengthen_it():
    uneven = [10.0, 40.0, 10.0, 5.0, 5.0, 5.0] + [1.0] * 6
    evened = [10.0, 20.0, 10.0, 12.0, 12.0, 12.0] + [1.0] * 6
    lumps = dict(lumps=[0, 2], held=[300, 20, 300, 60, 60, 60] + [10] * 6)
    gaps, _ = fit_gap([_ring(uneven, lumps=[]), _ring(evened, **lumps)], {})
    assert gaps == [pytest.approx(42.0), pytest.approx(22.0)], "through volume 1 both times, not the short way of the start"


def test_with_fewer_than_two_lumps_at_the_end_there_is_no_gap():
    even = [10.0] * 6 + [1.0] * 6
    gaps, held = fit_gap([_ring(even, lumps=[]), _ring(even, lumps=[2])], {})
    assert gaps == [None, None] and held == 0.0


def test_two_lump_volumes_joined_through_a_way_are_one_lump_with_one_centre():
    even = [10.0] * 6 + [1.0] * 6
    last = _ring(even, lumps=[0, 1, 3], held=[200, 250, 20, 300, 15, 15] + [10] * 6)
    gaps, _ = fit_gap([last, last], {})
    assert gaps[1] == pytest.approx(2 * 1.0 + 10.0), "from volume 1, the fuller of the joined pair, to volume 3"


def test_held_lumps_are_the_energy_in_lumps_two_thirds_through_that_is_in_lumps_at_the_end():
    even = [10.0] * 6 + [1.0] * 6
    then = _ring(even, lumps=[0, 2, 4], held=[300, 50, 250, 50, 250, 100] + [10] * 6)
    last = _ring(even, lumps=[0, 5], held=[550, 50, 0, 50, 50, 300] + [10] * 6)
    _, held = fit_gap([then, then, then, last], {2: 0})
    assert held == pytest.approx((300 + 250) / 1000), "volume 0 held, 2 fell into 0, 4 is no lump any more"
