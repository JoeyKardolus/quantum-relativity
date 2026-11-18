"""Near its source the law is 1 / distance; toward half the box the copies flatten it; the cube's field has the cube's symmetry only."""
import numpy as np
import pytest

from physics.scenes.along_grid.definitions.box_depth import box_depth
from physics.scenes.along_grid.definitions.box_field import box_field


def test_box_depth_halves_when_the_distance_doubles_near_the_source():
    near = [box_depth(radius, 500.0, 1000.0) for radius in (1.0, 2.0, 4.0)]
    assert near[0] / near[1] == pytest.approx(2.0, rel=0.02) and near[1] / near[2] == pytest.approx(2.0, rel=0.02)
    assert 0.0 < box_depth(400.0, 500.0, 1000.0) < 0.5 * (1.0 / 400.0 - 1.0 / 500.0), "the copies flatten the pull"
    with pytest.raises(ValueError):
        box_depth(0.0, 1.0, 10.0)


def test_the_two_forms_of_the_law_agree_along_an_axis():
    for across in (8, 16):
        field = box_field(across)
        for volumes in range(1, across // 2):
            summed = box_depth(float(volumes), across / 2.0, float(across))
            assert field[volumes, 0, 0] - field[across // 2, 0, 0] == pytest.approx(summed, rel=0.05)


def test_box_field_is_zero_farthest_away_alike_along_the_three_axes_and_not_alike_off_them():
    field = box_field(8)
    assert field[4, 4, 4] == 0.0 and field.shape == (8, 8, 8)
    assert field[3, 0, 0] == pytest.approx(field[0, 3, 0]) == pytest.approx(field[0, 0, 5])
    assert np.all(np.diff(field[1:5, 0, 0]) < 0.0), "deeper nearer the source"
    assert field[2, 2, 1] / field[3, 0, 0] < 0.97, "same straight-line distance, off the axis: the cube shows"
    with pytest.raises(ValueError):
        box_field(2)
