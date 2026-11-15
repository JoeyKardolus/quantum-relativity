"""Total energy is exact: 200 steps of the filled uneven start, every rule on, drift 0 quanta.

Every rule moves energy as whole quanta taken from one box and given to another:
transport, the leak, the clocks only scale how much. Runs on the Warp CPU device, so
it needs no GPU.
"""
from physics.fluid import Settings, new_grid, stats, step


def test_total_energy_does_not_change_by_one_quantum_in_200_steps():
    grid = new_grid(16, seed=1, unevenness=0.5, device="cpu")
    settings = Settings.all_on()
    start = stats(grid, settings)
    for _ in range(200):
        step(grid, settings)
    end = stats(grid, settings)
    assert end["steps"] == 200
    assert end["energy_drift"] == 0
    assert end["total_energy"] == start["total_energy"]
    assert end["heat"] > 0.0
