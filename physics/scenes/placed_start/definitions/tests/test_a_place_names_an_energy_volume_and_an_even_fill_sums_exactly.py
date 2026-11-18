"""A place gives the start's cell id only for an energy volume; an even fill is whole, level and exact."""
import pytest

from physics.scenes.placed_start.definitions.even_fill import even_fill
from physics.scenes.placed_start.definitions.place_cell import place_cell


def test_a_place_with_three_even_indices_is_its_raster_id():
    assert place_cell(8, (4, 4, 4)) == 292
    assert place_cell(8, (0, 2, 6)) == 22


@pytest.mark.parametrize("place", [(1, 0, 0), (0, 0, 3), (8, 0, 0), (-2, 0, 0)])
def test_a_joint_or_a_place_off_the_lattice_is_refused(place):
    with pytest.raises(ValueError):
        place_cell(8, place)


def test_an_even_fill_sums_exactly_and_no_two_boxes_differ_by_more_than_one():
    block = even_fill(1000)
    assert block.shape == (6, 6, 6) and int(block.sum()) == 1000
    assert int(block.max()) - int(block.min()) == 1
    assert (even_fill(216 * 5) == 5).all()
    with pytest.raises(ValueError):
        even_fill(-1)
