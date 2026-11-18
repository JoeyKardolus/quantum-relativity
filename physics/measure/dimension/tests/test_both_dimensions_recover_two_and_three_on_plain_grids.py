"""Proved first: on plain grids whose neighbours lie along one axis each, both dimensions come within the pass mark of 2 and 3."""
import pytest

from physics.measure.dimension import adjacency, grid_graph, measure_dimension, passes

GRIDS = {2: 101, 3: 41}


@pytest.mark.parametrize("dimensions", sorted(GRIDS))
def test_both_measures_give_the_grid_s_dimension_within_the_pass_mark(dimensions):
    measured = measure_dimension(adjacency(*grid_graph(GRIDS[dimensions], dimensions)))
    assert passes(measured["ball_dimension"], dimensions)
    assert passes(measured["spectral_dimension"], dimensions)
