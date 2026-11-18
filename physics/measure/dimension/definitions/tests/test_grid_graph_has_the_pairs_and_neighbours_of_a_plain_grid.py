"""The reference grid by counting: pairs and neighbours of a plain grid are known exactly."""
import numpy as np
import pytest

from physics.measure.dimension.definitions.adjacency import adjacency
from physics.measure.dimension.definitions.grid_graph import grid_graph


def _degrees(side, dimensions, wrapped, diagonals):
    low, high, node_count = grid_graph(side, dimensions, wrapped, diagonals)
    return np.asarray(adjacency(low, high, node_count).sum(axis=1)).ravel().reshape((side,) * dimensions)


def test_an_open_cubic_grid_has_three_n_squared_n_minus_one_pairs():
    low, high, node_count = grid_graph(5, 3)
    assert node_count == 125 and len(low) == len(high) == 3 * 25 * 4
    assert set(np.abs(high - low).tolist()) == {1, 5, 25}


@pytest.mark.parametrize("dimensions, diagonals, neighbours", [(2, False, 4), (3, False, 6), (2, True, 8), (3, True, 26)])
def test_every_node_of_a_wrapped_grid_has_the_same_number_of_neighbours(dimensions, diagonals, neighbours):
    assert set(_degrees(5, dimensions, True, diagonals).ravel().tolist()) == {float(neighbours)}


def test_an_open_grid_with_diagonals_has_fewer_neighbours_at_its_faces():
    degrees = _degrees(3, 2, False, True)
    assert degrees.tolist() == [[3, 5, 3], [5, 8, 5], [3, 5, 3]]


def test_a_wrapped_grid_of_side_two_with_diagonals_joins_every_node_to_every_other():
    assert set(_degrees(2, 3, True, True).ravel().tolist()) == {7.0}


@pytest.mark.parametrize("side, dimensions", [(1, 3), (4, 0)])
def test_a_side_under_two_or_no_dimension_is_refused(side, dimensions):
    with pytest.raises(ValueError):
        grid_graph(side, dimensions)
