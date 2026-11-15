"""Total energy is exact: 200 steps of an uneven ball, all rules on, drift 0 quanta."""
from physics.fluid import Settings, new_grid, stats, step
from physics.fluid.tests._gpu import needs_gpu


@needs_gpu
def test_total_energy_does_not_change_by_one_quantum_in_200_steps():
    grid = new_grid(24, seed=1, unevenness=0.5)
    settings = Settings.all_on()
    start = stats(grid, settings)
    for _ in range(200):
        step(grid, settings)
    end = stats(grid, settings)
    assert end["steps"] == 200
    assert end["energy_drift"] == 0
    assert end["total_energy"] == start["total_energy"]
