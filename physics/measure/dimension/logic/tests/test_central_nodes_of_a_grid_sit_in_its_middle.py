"""Central nodes on graphs whose middle is known."""
import numpy as np
import pytest

from physics.measure.dimension.definitions.adjacency import adjacency
from physics.measure.dimension.definitions.grid_graph import grid_graph
from physics.measure.dimension.logic.central_nodes import central_nodes


def test_the_most_central_node_of_a_square_grid_is_within_one_step_of_its_middle():
    picked = central_nodes(adjacency(*grid_graph(21, 2)), 1)
    row, column = np.unravel_index(picked[0], (21, 21))
    assert abs(row - 10) <= 1 and abs(column - 10) <= 1


def test_the_most_central_node_of_a_line_is_its_middle_node():
    assert central_nodes(adjacency(*grid_graph(31, 1)), 1).tolist() == [15]


def test_only_the_largest_connected_piece_is_searched_and_asking_for_more_than_it_holds_is_refused():
    matrix = adjacency(np.array([0, 1, 3]), np.array([1, 2, 4]), 5)
    assert set(central_nodes(matrix, 3).tolist()) == {0, 1, 2}
    with pytest.raises(ValueError):
        central_nodes(matrix, 4)
