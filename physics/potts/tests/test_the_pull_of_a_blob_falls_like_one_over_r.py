"""The pull potential of a single blob of A falls like -G M / r outside it, within 5%."""
import numpy as np

from physics.potts import ENERGY_VOLUME, QUANTUM, Settings, Universe, cell_sums, pull
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_the_potential_outside_a_blob_is_minus_g_m_over_r():
    side, radius = 48, 5.0
    axis = np.arange(side) - (side - 1) / 2.0
    distance = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)
    inside = distance <= radius
    owner = np.where(inside, 0, -1)
    universe = Universe.from_host(owner, np.where(inside, 1.0, 0.0), np.array([ENERGY_VOLUME]), 2, seed=1)
    settings = Settings.all_on(potential_cycles=12)
    cell_sums(universe, settings)
    pull(universe, settings)
    potential = universe.potential.numpy()
    mass = int(universe.energy.numpy().sum()) * QUANTUM
    outside = (distance > radius + 1.5) & (distance < side / 2.0 - 2.0)
    expected = -settings.attraction * mass / distance[outside]
    assert outside.sum() > 1000
    assert np.max(np.abs(potential[outside] / expected - 1.0)) < 0.05
