"""Validation first (bible "How results are trusted"): on plain grids, whose true dimension is
known, the ball and the spectral dimension both come within 10% of 2 and 3."""
import numpy as np
import pytest
from scipy.sparse import csgraph

from physics.measure import adjacency, ball_dimension, central_nodes, grid_graph, spectral_dimension

GRIDS = {2: 101, 3: 41}


def _measured(dimensions):
    low, high, node_count = grid_graph(GRIDS[dimensions], dimensions)
    matrix = adjacency(low, high, node_count)
    sources = central_nodes(matrix, 8)
    reach = int(csgraph.shortest_path(matrix, unweighted=True, indices=sources[0]).max()) // 4
    ball = ball_dimension(matrix, sources, np.arange(2, reach + 1))
    spectral = spectral_dimension(matrix, sources, np.arange(reach ** 2 // 10, reach ** 2 + 1))
    return ball["dimension"], spectral["dimension"]


@pytest.mark.parametrize("dimensions", sorted(GRIDS))
def test_both_measures_give_the_grids_dimension_within_ten_percent(dimensions):
    ball, spectral = _measured(dimensions)
    assert abs(ball - dimensions) <= 0.1 * dimensions
    assert abs(spectral - dimensions) <= 0.1 * dimensions


def test_the_central_nodes_of_a_grid_sit_in_its_middle():
    low, high, node_count = grid_graph(21, 2)
    sources = central_nodes(adjacency(low, high, node_count), 1)
    row, column = np.unravel_index(sources[0], (21, 21))
    assert abs(row - 10) <= 1 and abs(column - 10) <= 1


def test_a_pair_listed_twice_counts_once_and_a_self_pair_is_refused():
    matrix = adjacency(np.array([0, 1, 0]), np.array([1, 0, 2]), 3)
    assert matrix.toarray().tolist() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    with pytest.raises(ValueError):
        adjacency(np.array([1]), np.array([1]), 3)
