"""Region contrast is the rms of (block energy / mean block energy - 1) over blocks of 2 x 2 x 2 volumes, 0 when even."""
import numpy as np
import pytest

from physics.fit.measures.definitions.region_contrast import region_contrast


def test_an_even_field_has_no_contrast_however_uneven_it_is_inside_a_block():
    even = np.full((4, 4, 4), 100)
    assert region_contrast(even) == 0.0
    inside = even.copy()
    inside[0, 0, 0], inside[1, 1, 1] = 150, 50
    assert region_contrast(inside) == 0.0, "the block still holds 800"


def test_one_block_of_eight_holding_double_gives_a_known_contrast():
    field = np.full((4, 4, 4), 100)
    field[:2, :2, :2] = 200
    assert region_contrast(field) == pytest.approx(np.sqrt(7.0) / 9.0), "one block at 16/9 of the mean, seven at 8/9"


def test_a_side_that_is_odd_or_under_four_or_an_empty_field_has_no_blocks_to_compare():
    assert region_contrast(np.arange(8.0).reshape(2, 2, 2)) == 0.0
    assert region_contrast(np.arange(27.0).reshape(3, 3, 3)) == 0.0
    assert region_contrast(np.zeros((4, 4, 4))) == 0.0
