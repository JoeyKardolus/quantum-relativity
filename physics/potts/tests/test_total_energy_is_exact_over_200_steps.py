"""Total energy is exact: 200 steps of the uneven start, every rule on, drift 0 quanta.

The flow takes whole quanta from one box and gives them to another, a box move hands
its energy on exactly, and a collapse only relabels boxes.
"""
from physics.potts import Settings, new_universe, stats, step
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_total_energy_does_not_change_by_one_quantum_in_200_steps():
    universe = new_universe(6, 4, seed=2, unevenness=0.5)
    settings = Settings.all_on()
    start = stats(universe, settings)
    for _ in range(200):
        step(universe, settings)
    end = stats(universe, settings)
    assert universe.steps == 200
    assert end["energy_drift"] == 0
    assert end["total_energy"] == start["total_energy"]
