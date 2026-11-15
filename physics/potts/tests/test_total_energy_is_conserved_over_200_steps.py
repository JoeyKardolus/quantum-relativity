"""Total energy is conserved: 200 steps of the uneven start, every rule on, drift only float64 rounding.

Total energy is the sum of |psi|**2 over the boxes. Every hop and on-site turn of the
energy fluid is an exact unitary rotation (hop.py, on_site.py), a box move hands a box's
|psi|**2 on whole (box_moves.py), collapse and split only relabel. The equation's own
fluid energy (flow, pull, push) is printed beside it: it is not a rule and is not
asserted (the pull, the cells and friction change it).
"""
from physics.potts import Settings, new_universe, stats, step
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_total_energy_moves_by_no_more_than_rounding_in_200_steps():
    universe = new_universe(7, 4, seed=2, unevenness=0.5)
    settings = Settings.all_on()
    start = stats(universe, settings)
    for _ in range(200):
        step(universe, settings)
    end = stats(universe, settings)
    print(f"total energy {start['total_energy']:.12f} -> {end['total_energy']:.12f} (relative "
          f"{end['relative_drift']:.2e}); fluid energy {start['fluid_energy']:.3f} -> {end['fluid_energy']:.3f}; "
          f"heat {end['heat']:.3f}")
    assert universe.steps == 200
    assert abs(end["relative_drift"]) < 1e-10
