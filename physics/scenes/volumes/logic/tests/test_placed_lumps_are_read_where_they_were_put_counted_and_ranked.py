"""On a placed start the field holds each lump at its place, the lumps are counted, joined ones as one region, and the fullest come first."""
import numpy as np

from physics.engine import Settings
from physics.scenes.placed_start import packed_start
from physics.scenes.tests._gpu import needs_gpu
from physics.scenes.volumes.logic.count_lumps import count_lumps
from physics.scenes.volumes.logic.observer_cells import observer_cells
from physics.scenes.volumes.logic.volume_field import volume_field


@needs_gpu
def test_three_placed_lumps_two_of_them_neighbours():
    universe = packed_start(8, {(4, 4, 4): 4.0, (4, 4, 6): 3.0, (0, 0, 0): 2.0})
    field, left_out = volume_field(universe)
    assert field.shape == (4, 4, 4) and left_out == 0
    assert field[2, 2, 2] == 4 * 648000 and field[2, 2, 3] == 3 * 648000 and field[0, 0, 0] == 2 * 648000
    assert count_lumps(universe, Settings()) == {"lumps": 3, "largest_region": 2}
    assert observer_cells(universe, 3).tolist() == [292, 294, 0]
    assert observer_cells(universe, 0).size == 0 and observer_cells(universe, 1000).size == 64


@needs_gpu
def test_an_even_grid_has_no_lump():
    assert count_lumps(packed_start(8, {}), Settings()) == {"lumps": 0, "largest_region": 0}
