"""A distance along a row is half the first cell, the whole cells between and half the last; the volumes' pull is the engine's at their ids."""
import numpy as np
import pytest

from physics.scenes.along_grid.definitions.row_cells import row_cells
from physics.scenes.along_grid.logic.middle_distances import middle_distances
from physics.scenes.along_grid.logic.volume_pull import volume_pull
from physics.scenes.placed_start import packed_start
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_the_distance_to_the_next_volume_is_half_the_core_the_joint_and_half_that_volume():
    universe = packed_start(8, {(4, 4, 4): 4.0})
    length = universe.host("length").astype(np.float64)
    row = row_cells(8, 292, 0, 1, 2)
    found = middle_distances(universe, row)
    assert found[0] == 0.0
    assert found[1] == pytest.approx(0.5 * 2.5 + 0.5 * length[row[1]])
    assert found[2] == pytest.approx(0.5 * 2.5 + length[row[1]] + 0.5 * length[row[2]])
    assert length[row[1]] > 0.0 and 9.5 < length[row[2]] < 10.5, "a joint is as long as the balance makes it, a volume of 3000 a box 10"


@needs_gpu
def test_the_volumes_pull_is_the_engines_pull_at_the_even_lattice_places():
    universe = packed_start(8, {})
    pull = universe.host("pull")
    grid = volume_pull(universe)
    assert grid.shape == (4, 4, 4) and grid[2, 2, 2] == pull[292] and grid[0, 1, 3] == pull[22]
