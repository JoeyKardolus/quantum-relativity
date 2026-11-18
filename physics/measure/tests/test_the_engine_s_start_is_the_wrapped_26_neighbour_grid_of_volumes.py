"""Axiom 3 on the measure: at the start every energy volume has a joint to each of its 26
neighbours, so the volume graph read off the engine is exactly the wrapped 26-neighbour grid."""
import numpy as np
import pytest

from physics.engine import ENERGY_VOLUME, new_universe, read_cells, read_ways
from physics.measure import measure_dimension, volume_graph


@pytest.mark.parametrize("cells_per_side", [8, 16])
def test_the_volume_graph_of_a_fresh_universe_equals_the_reference_grid(cells_per_side, device, lattice_graph):
    universe = new_universe(cells_per_side=cells_per_side, seed=1, device=device)
    kinds = read_cells(universe)["kind"]
    matrix, volumes = volume_graph(read_ways(universe), kinds)
    assert volumes.tolist() == np.flatnonzero(kinds == ENERGY_VOLUME).tolist()
    assert len(volumes) == (cells_per_side // 2) ** 3
    assert (matrix != lattice_graph(cells_per_side // 2)).nnz == 0
    assert set(np.asarray(matrix.sum(axis=1)).ravel().tolist()) == {26.0}


def test_the_fresh_universe_and_its_reference_measure_the_same_dimensions(device, lattice_graph):
    universe = new_universe(cells_per_side=16, seed=1, device=device)
    matrix, _ = volume_graph(read_ways(universe), read_cells(universe)["kind"])
    measured, reference = measure_dimension(matrix), measure_dimension(lattice_graph(8))
    assert measured == reference
    assert measured["farthest"] == 4 and measured["radii"] == [2, 3] and measured["shells"] == [98.0, 218.0]
