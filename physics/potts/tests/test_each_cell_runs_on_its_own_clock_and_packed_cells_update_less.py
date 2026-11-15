"""Local clocks: each cell takes its own substeps; a packed cell does fewer, longer updates.

Bible: time is local, the more packed the slower. In an even start at rest every cell's
fastest flow is its local sound speed times its clock; one energy volume holding 8
times its energy has a slower clock (its sound is faster, sqrt 8, but its clock slower,
2 / 9), so time_levels gives it a lower level (fewer substeps of its own) than the even
cells. Over 5 steps of the uneven start the box updates made are fewer than every box
updating at the finest substep would make, and the total energy holds to rounding.
"""
import numpy as np

from physics.potts import Settings, cell_sums, new_universe, stats, step, time_levels
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import box_energy, cell_id, even_universe, set_box_energy


@needs_gpu
def test_a_packed_cell_gets_a_lower_time_level_and_boxes_update_at_their_own_level():
    universe = even_universe(7, 4)
    packed = cell_id(7, 3, 3, 3)
    owner = universe.owner.numpy()
    energy = box_energy(universe)
    energy[owner == packed] *= 8
    set_box_energy(universe, energy)
    settings = Settings.all_on()
    cell_sums(universe, settings)
    finest = time_levels(universe, settings)
    levels = universe.cell_level.numpy()
    others = np.delete(levels, packed)
    print(f"packed cell level {levels[packed]}, others {np.unique(others)}, finest {finest}")
    assert levels[packed] < others.min()
    uneven = new_universe(7, 4, seed=1, unevenness=0.3)
    for _ in range(5):
        step(uneven, settings)
    end = stats(uneven, settings)
    print(f"box updates {end['box_updates']}, finest substeps {end['substeps']} x occupied {end['occupied']}")
    assert end["box_updates"] < end["substeps"] * end["occupied"]
    assert abs(end["relative_drift"]) < 1e-10
