"""The pull potential of a single blob falls like -G M / r outside it, within 5%."""
import numpy as np

from physics.fluid import QUANTUM, Grid, Settings, pull_potential
from physics.fluid.tests._gpu import needs_gpu


@needs_gpu
def test_the_potential_outside_a_blob_is_minus_g_m_over_r():
    side, radius = 48, 5.0
    axis = np.arange(side) - (side - 1) / 2.0
    distance = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)
    a = np.where(distance <= radius, 1.0, 0.0)
    grid = Grid.from_host(a, np.zeros_like(a))
    settings = Settings.all_on()
    pull_potential(grid, settings, cycles=12)
    potential = grid.potential.numpy()
    mass = int(grid.a_energy.numpy().sum()) * QUANTUM
    outside = (distance > radius + 1.5) & (distance < side / 2.0 - 2.0)
    expected = -settings.attraction * mass / distance[outside]
    assert outside.sum() > 1000
    assert np.max(np.abs(potential[outside] / expected - 1.0)) < 0.05
