"""Energy volumes start about 10 Planck boxes across on average, sizes following each one's energy (owner, 2026-11-16).

new_universe() (the defaults: 7 cells of 10 boxes a side, seed 1, unevenness 0.3); the
first step runs the start's size correction, which lets every cell's size follow its own
energy (no set size, no minimum). After it the energy volumes' mean width (the cube root
of each one's box count, stats volume_across) is 10 +- 2 boxes, and their sizes are not
all one: more energy, fewer boxes (rule 4), so box count and energy are anticorrelated.
"""
import numpy as np

from physics.potts import ENERGY_VOLUME, Settings, new_universe, stats, step
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_energy_volumes_start_10_plus_minus_2_boxes_across_and_follow_their_energy():
    universe = new_universe()
    settings = Settings.all_on()
    step(universe, settings)
    now = stats(universe, settings)
    volumes = (universe.kind.numpy() == ENERGY_VOLUME) & (universe.cell_boxes.numpy() > 0)
    boxes = universe.cell_boxes.numpy()[volumes]
    energy = universe.cell_energy.numpy()[volumes]
    widths = np.cbrt(boxes)
    correlation = float(np.corrcoef(boxes, energy)[0, 1])
    print(f"energy volumes {volumes.sum()}, across {now['volume_across']:.2f} (min {widths.min():.2f}, "
          f"max {widths.max():.2f}), box count vs energy correlation {correlation:.2f}")
    assert 8.0 <= now["volume_across"] <= 12.0
    assert widths.max() > widths.min()
    assert correlation < 0
