"""Local clocks read slower where energy is packed (bible: time is local, each region's clock set by its concentration).

An even start (7 cells of 4 boxes a side, every box holding 1); one energy volume is
given 8 times its energy per box, taken evenly from every other occupied box, so total
energy is unchanged. After 5 steps every living cell's clock reading (universe.cell_time,
the proper time it lived, local_time.py) is read: the packed cell's reading is the
lowest of all, and over every cell the reading falls as the energy per box rises (rank
correlation below 0).
"""
import numpy as np
from scipy import stats as scipy_stats

from physics.potts import Settings, stats, step
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import box_energy, cell_id, even_universe, set_box_energy


@needs_gpu
def test_the_packed_cell_reads_the_least_proper_time_and_readings_fall_with_packing():
    universe = even_universe(7, 4)
    packed = cell_id(7, 3, 3, 3)
    owner = universe.owner.numpy()
    energy = box_energy(universe)
    extra = 7.0 * energy[owner == packed].sum()
    energy[owner == packed] *= 8.0
    others = (owner >= 0) & (owner != packed)
    energy[others] -= extra / others.sum()
    assert np.all(energy[others] > 0)
    set_box_energy(universe, energy)
    settings = Settings.all_on()
    for _ in range(5):
        step(universe, settings)
    now = stats(universe, settings)
    living = (universe.alive.numpy() == 1) & (universe.cell_boxes.numpy() > 0)
    reading = universe.cell_time.numpy()
    packing = universe.cell_energy.numpy() / np.maximum(universe.cell_boxes.numpy(), 1)
    rank = scipy_stats.spearmanr(packing[living], reading[living]).statistic
    print(f"packed {reading[packed]:.3f}, mean {reading[living].mean():.3f}, "
          f"range {now['local_time_min']:.3f}..{now['local_time_max']:.3f}, rank correlation {rank:.2f}")
    assert reading[packed] == reading[living].min()
    assert rank < 0
