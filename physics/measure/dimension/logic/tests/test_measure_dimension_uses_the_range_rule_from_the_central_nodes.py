"""The whole measurement on one square grid: its ranges are the range rule's, its numbers the two measures'."""
from physics.measure.dimension.definitions.adjacency import adjacency
from physics.measure.dimension.definitions.grid_graph import grid_graph
from physics.measure.dimension.definitions.pass_mark import passes
from physics.measure.dimension.logic.measure_dimension import measure_dimension


def test_a_square_grid_of_101_a_side_is_measured_out_to_25_steps_and_gives_two():
    measured = measure_dimension(adjacency(*grid_graph(101, 2)))
    assert measured["farthest"] == 100 and measured["nodes"] == 101 * 101
    assert measured["radii"] == list(range(2, 26))
    assert (measured["times"][0], measured["times"][-1]) == (62, 625)
    assert len(measured["shells"]) == 24 and len(measured["return_probability"]) == 564
    assert passes(measured["ball_dimension"], 2) and passes(measured["spectral_dimension"], 2)
