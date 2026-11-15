"""Joints are full cells: they hold energy like energy volumes and start at comparable sizes.

Bible rule 8 (owner, 2026-11-15): a joint is a group of Planck boxes whose size follows
its own energy, no longer a thin wall. new_universe gives every box, joint or volume,
the same field, so the mean box of each kind holds about 1; one size law (one
cell_rest) then gives each cell a want of k**3 times its own ripple, and after the
start's size correction the mean joint and the mean energy volume hold comparable box
counts. The margin is at least as wide as the filled cube.
"""
import numpy as np
import pytest

from physics.potts import ENERGY_VOLUME, JOINT, QUANTUM, Settings, cell_sums, new_universe, size_correction, stats
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_both_kinds_hold_the_same_energy_per_box_and_start_at_comparable_sizes():
    universe = new_universe(6, 4, seed=1, unevenness=0.3)
    filled = 6 * 4
    assert (universe.side - filled) // 2 >= filled
    owner = universe.owner.numpy()
    energy = universe.energy.numpy() * QUANTUM
    kind = universe.kind.numpy()
    box_kind = kind[owner[owner >= 0]]
    per_box = energy[owner >= 0]
    assert per_box[box_kind == JOINT].mean() == pytest.approx(1.0, rel=0.1)
    assert per_box[box_kind == ENERGY_VOLUME].mean() == pytest.approx(1.0, rel=0.1)
    settings = Settings.all_on()
    cell_sums(universe, settings)
    want = universe.cell_want.numpy()
    held = universe.cell_energy.numpy() * QUANTUM
    expected = universe.base_boxes * (universe.cell_rest / held) ** settings.size_power
    assert np.allclose(want, expected, rtol=1e-4)
    size_correction(universe, settings)
    ratio = stats(universe, settings)["size_ratio"]
    print(f"mean joint boxes / mean energy-volume boxes after the correction: {ratio:.3f}")
    assert 0.8 < ratio < 1.25
