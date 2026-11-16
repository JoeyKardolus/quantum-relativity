"""Friction acts only on flow crossing into a joint: a fluid at rest makes (next to) no heat, and never negative heat.

An even start (6 cells of 4 boxes a side, seed 1, unevenness 0: every box holds 1,
phase 0), every fluid rule on (Settings.all_on(): the push, stiffer in joints, so
energy volumes' and joints' phases turn at different on-site rates; the pull, felt by
the energy volumes; friction at the seams), the foam held as it is (cell sums, pull,
energy flow each step; box moves stir the fluid, so they are left out). The phase step
the on-site rates build at an A-B face carries no flow behind it, and friction leaves
it (hop.py). Over 200 steps the heat never falls below 0 at any step and stays under
0.1% of the clocked fluid energy; what it books is the pull's onset (the potential
grows from 0 over the first steps and sets the A-fluid moving). Measured 2026-11-16:
66 of 166,000 (old friction, which relaxed the whole phase step: 44 with the pull felt
by every box, 85,000 with box moves on).
"""
from physics.potts import Settings, cell_sums, energy_flow, fluid_energy, new_universe, pull
from physics.potts.region_host import region_host
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_a_fluid_at_rest_books_no_heat_and_never_negative_heat():
    settings = Settings.all_on()
    universe = new_universe(6, 4, seed=1, unevenness=0.0)
    universe.corrected = True
    lowest = 0.0
    for count in range(200):
        cell_sums(universe, settings)
        pull(universe, settings)
        cell_sums(universe, settings)
        energy_flow(universe, settings)
        lowest = min(lowest, float(region_host(universe, universe.heat).min()))
        if count % 50 == 49:
            heat = float(region_host(universe, universe.heat).sum())
            print(f"step {count + 1}: heat {heat:.4f}, lowest box heat {lowest:.2e}")
    heat = float(region_host(universe, universe.heat).sum())
    energy = fluid_energy(universe, settings, clocked=True)
    print(f"heat {heat:.4f} of clocked fluid energy {energy:.1f}")
    assert lowest >= 0.0
    assert heat <= 1e-3 * energy
