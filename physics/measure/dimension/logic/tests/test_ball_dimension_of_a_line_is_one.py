"""Ball dimension where shells can be counted by hand."""
import numpy as np
import pytest

from physics.measure.dimension.definitions.adjacency import adjacency
from physics.measure.dimension.definitions.grid_graph import grid_graph
from physics.measure.dimension.logic.ball_dimension import ball_dimension


def test_on_a_line_every_shell_holds_two_nodes_so_the_dimension_is_exactly_one():
    measured = ball_dimension(adjacency(*grid_graph(101, 1)), [50], np.arange(2, 20))
    assert measured["shells"] == [2.0] * 18
    assert measured["dimension"] == pytest.approx(1.0, abs=1e-12)


def test_on_a_square_grid_the_shell_at_r_steps_holds_four_r_nodes():
    measured = ball_dimension(adjacency(*grid_graph(41, 2)), [20 * 41 + 20], np.arange(2, 11))
    assert measured["shells"] == [4.0 * radius for radius in range(2, 11)]
    assert measured["dimension"] == pytest.approx(2.0, abs=1e-12)


def test_a_radius_past_the_graph_s_end_and_too_few_radii_are_refused():
    matrix = adjacency(*grid_graph(5, 1))
    with pytest.raises(ValueError):
        ball_dimension(matrix, [2], [2, 3])
    with pytest.raises(ValueError):
        ball_dimension(matrix, [2], [2])
