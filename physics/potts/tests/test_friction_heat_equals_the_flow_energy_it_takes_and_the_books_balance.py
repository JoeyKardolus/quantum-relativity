"""Friction makes no energy: the heat it books is the flow energy it took, and the fluid's books balance.

Friction (hop.py) turns the two boxes of an energy-volume/joint face towards each
other, which changes the flow energy of every face those boxes touch; the heat booked
is that whole drop. The other phases of a step (the pull's re-solve, the joints' new
compression, box moves) change the fluid's energy too, by the theory's own rules; step.py
tallies each as work, so energy_balance = clocked fluid energy + heat - work moves only
by the fluid steps' own splitting error. A small uneven start (4 cells of 4 boxes a
side, seed 1, unevenness 0.3), every rule on, flat clocks (clock_scale 1e9), where the
rules say the fluid's energy is kept while nothing else changes:
  - frozen: after 3 steps the whole fluid is set flowing along x (its phase turned 0.3
    more per box), then 4 runs of energy_flow alone at courant 0.05; the clocked fluid
    energy falls by the heat booked within 1% of it (booking only the rubbed face
    misses the change on the boxes' other faces, which a flowing fluid has);
  - stepped: 8 whole steps at courant 0.0125; every step books heat and the balance
    moves by at most 5% of that step's heat (the first step, right after the start's
    size correction, has the largest splitting error).
"""
import numpy as np

from physics.potts import Settings, cell_sums, energy_flow, fluid_energy, new_universe, stats, step
from physics.potts.region_host import region_host
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_frozen_fluid_loses_exactly_the_heat_friction_books():
    settings = Settings.all_on(clock_scale=1e9, courant=0.05)
    universe = new_universe(4, 4, seed=1, unevenness=0.3)
    for _ in range(3):
        step(universe, settings)
    psi = universe.psi.numpy()
    turn = 0.3 * np.arange(universe.side)[:, None, None]
    universe.psi.assign(np.stack([psi[..., 0] * np.cos(turn) - psi[..., 1] * np.sin(turn),
                                  psi[..., 0] * np.sin(turn) + psi[..., 1] * np.cos(turn)], axis=-1))
    cell_sums(universe, settings)
    start = fluid_energy(universe, settings, clocked=True)
    heat_start = float(region_host(universe, universe.heat).sum())
    for _ in range(4):
        energy_flow(universe, settings)
    lost = start - fluid_energy(universe, settings, clocked=True)
    heat = float(region_host(universe, universe.heat).sum()) - heat_start
    print(f"fluid energy lost {lost:.4f}, heat booked {heat:.4f}")
    assert heat > 0
    assert abs(lost - heat) <= 0.01 * heat


@needs_gpu
def test_every_step_the_balance_moves_by_a_small_share_of_the_heat():
    settings = Settings.all_on(clock_scale=1e9, courant=0.0125)
    universe = new_universe(4, 4, seed=1, unevenness=0.3)
    before = stats(universe, settings)
    for count in range(8):
        step(universe, settings)
        after = stats(universe, settings)
        heat = after["heat"] - before["heat"]
        moved = after["energy_balance"] - before["energy_balance"]
        print(f"step {count + 1}: heat {heat:.4f}, balance moved {moved:.3e}")
        assert heat > 0
        assert abs(moved) <= 0.05 * heat
        assert abs(after["relative_drift"]) < 1e-12
        before = after
