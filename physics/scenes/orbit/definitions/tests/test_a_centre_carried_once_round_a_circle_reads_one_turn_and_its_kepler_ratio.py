"""A centre taken once round a circle of radius 2 in 100 time units reads one turn and 100 squared over 2 cubed."""
import numpy as np
import pytest

from physics.scenes.orbit.definitions.blob_centre import blob_centre
from physics.scenes.orbit.definitions.went_round import went_round

ANGLES = np.linspace(0.0, 2.0 * np.pi, 41)
CIRCLE = np.stack([2.0 * np.cos(ANGLES), 2.0 * np.sin(ANGLES), np.zeros(41)], axis=1)
TIMES = np.linspace(0.0, 100.0, 41).tolist()


def test_one_whole_turn_and_keplers_ratio():
    found = went_round(TIMES, [10.0] * 41, CIRCLE)
    assert found["turned"] == pytest.approx(1.0) and found["lost_sample"] == -1
    assert found["period_squared_over_radius_cubed"] == pytest.approx(100.0 ** 2 / 8.0)
    assert found["radius"] == pytest.approx([2.0] * 41)


def test_the_turn_stops_counting_when_half_the_energy_is_lost_and_the_other_way_round_is_negative():
    found = went_round(TIMES, [10.0] * 20 + [4.0] * 21, CIRCLE)
    assert found["lost_sample"] == 20 and found["turned"] == pytest.approx(0.5)
    assert went_round(TIMES, [10.0] * 41, CIRCLE[::-1])["turned"] == pytest.approx(-1.0)
    still = went_round([0.0, 1.0], [5.0, 5.0], [[2.0, 0.0, 0.0]] * 2)
    assert still["turned"] == 0.0 and still["period_squared_over_radius_cubed"] == 0.0
    with pytest.raises(ValueError):
        went_round([0.0], [1.0, 1.0], [[1.0, 0.0, 0.0]])


def test_the_blob_is_the_extra_energy_outside_the_big_volume_and_its_26_neighbours():
    before = np.full((8, 8, 8), 100.0)
    field = before.copy()
    field[4, 4, 4] += 700.0
    field[5, 5, 4] += 50.0
    field[6, 4, 4] += 30.0
    field[1, 4, 4] -= 80.0
    at_small, centre = blob_centre(field, before, 4, (6, 4, 4))
    assert at_small == 30.0 and centre == [2.0, 0.0, 0.0], "only the volume two away counts"
    assert blob_centre(before, before, 4, (6, 4, 4)) == (0.0, [0.0, 0.0, 0.0])
