"""Friction destroys no energy: the directed-flow energy rub takes is the heat it makes, and heat moves with energy.

Hand-built boxes, A flowing through B at rest: after one rub of dt the flow energy
lost, (1/2)|p|^2/e summed over both kinds, equals the heat made (to float precision)
and the box's momentum is kept; in a sealed box all the relative flow becomes heat.
Then transport carries that heat with the energy it sits in, the total unchanged.
"""
import numpy as np
import pytest

from physics.fluid import QUANTUM, Grid, Settings
from physics.fluid.rub import rub
from physics.fluid.transport import transport


def _flow_energy(grid: Grid) -> float:
    total = 0.0
    for momentum, quanta in ((grid.a_momentum, grid.a_energy), (grid.b_momentum, grid.b_energy)):
        energy = quanta.numpy() * QUANTUM
        squared = (momentum.numpy().astype(np.float64) ** 2).sum(axis=-1)
        total += float(0.5 * np.divide(squared, energy, out=np.zeros_like(energy), where=energy > 0).sum())
    return total


@pytest.mark.parametrize("compression", [1.0, 6.0])
def test_the_flow_energy_rub_takes_is_the_heat_it_makes(compression):
    settings = Settings.all_on()
    a = np.zeros((6, 6, 6))
    a[1:5, 1:5, 1:5] = 1.0
    b = np.where(a > 0, compression * settings.b_rest, 0.0)
    velocity = np.zeros((6, 6, 6, 3))
    velocity[1:5, 1:5, 1:5, 0] = 0.3
    grid = Grid.from_host(a, b, velocity, None, device="cpu")
    grid.start_a_mean = 1.0
    flow_before = _flow_energy(grid)
    momentum_before = grid.a_momentum.numpy().sum(axis=(0, 1, 2)) + grid.b_momentum.numpy().sum(axis=(0, 1, 2))
    rub(grid, settings, 0.05)
    heat = float(grid.heat.numpy().astype(np.float64).sum())
    assert heat > 0.0
    assert flow_before - _flow_energy(grid) == pytest.approx(heat, rel=1e-4)
    momentum_after = grid.a_momentum.numpy().sum(axis=(0, 1, 2)) + grid.b_momentum.numpy().sum(axis=(0, 1, 2))
    assert np.allclose(momentum_after, momentum_before, rtol=1e-5)
    if compression >= settings.seal_compression:
        assert _flow_energy(grid) == pytest.approx(0.5 * 64 * 0.09 / (1.0 + compression * settings.b_rest),
                                                   rel=1e-4)
    transport(grid, settings, 0.5)
    assert float(grid.heat.numpy().astype(np.float64).sum()) == pytest.approx(heat, rel=1e-5)
    assert grid.heat.numpy()[5, 1:5, 1:5].sum() > 0.0
