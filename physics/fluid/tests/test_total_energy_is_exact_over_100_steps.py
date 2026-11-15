"""Total energy is exact: 100 steps of the filled uneven start, all rules on, drift 0 quanta.

Runs on the Warp CPU device, so it needs no GPU.
"""
from physics.fluid import Settings, new_grid, stats, step


def test_total_energy_does_not_change_by_one_quantum_in_100_steps():
    grid = new_grid(16, seed=1, unevenness=0.5, device="cpu")
    settings = Settings.all_on()
    start = stats(grid, settings)
    for _ in range(100):
        step(grid, settings)
    end = stats(grid, settings)
    assert end["steps"] == 100
    assert end["energy_drift"] == 0
    assert end["total_energy"] == start["total_energy"]
