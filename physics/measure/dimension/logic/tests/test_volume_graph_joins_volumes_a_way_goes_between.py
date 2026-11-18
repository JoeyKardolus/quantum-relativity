"""The volume graph on hand-written engine rows: ways `(slot, a, b, joint)` and kinds per cell."""
import numpy as np
import pytest

from physics.engine import ENERGY_VOLUME, JOINT, NO_CELL
from physics.measure.dimension.logic.volume_graph import volume_graph

KINDS = np.array([ENERGY_VOLUME, JOINT, ENERGY_VOLUME, JOINT, ENERGY_VOLUME, NO_CELL])


def test_two_ways_between_the_same_volumes_count_once_and_a_way_inside_one_volume_joins_nothing():
    ways = np.array([[0, 0, 2, 1], [1, 2, 4, 3], [2, 0, 2, 3], [5, 4, 4, 1]])
    matrix, volumes = volume_graph(ways, KINDS)
    assert volumes.tolist() == [0, 2, 4]
    assert matrix.toarray().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]


def test_no_way_at_all_gives_volumes_that_touch_nothing():
    matrix, volumes = volume_graph(np.zeros((0, 4), dtype=np.int32), KINDS)
    assert volumes.tolist() == [0, 2, 4] and matrix.nnz == 0


@pytest.mark.parametrize("way", [[0, 0, 1, 3], [0, 0, 5, 1], [0, 0, 9, 1]])
def test_a_way_ending_on_a_joint_a_dead_cell_or_no_cell_is_refused(way):
    with pytest.raises(ValueError):
        volume_graph(np.array([way]), KINDS)


def test_a_grid_without_energy_volumes_is_refused():
    with pytest.raises(ValueError):
        volume_graph(np.zeros((0, 4), dtype=np.int32), np.array([JOINT, NO_CELL]))
