"""Friction makes no energy: the heat it books is the flow energy it took, and the fluid's books balance.

Friction (hop.py) turns the two boxes of an energy-volume/joint face towards each
other, which changes the flow energy of every face those boxes touch; the heat booked
is that whole drop, worked out exactly per pair. The other phases of a step (the pull's re-solve, the joints' new
compression, box moves) change the fluid's energy too, by the theory's own rules; step.py
tallies each as work, so energy_balance = clocked fluid energy + heat - work moves only
by the fluid steps' own splitting error. A small uneven start (4 cells of 4 boxes a
side, seed 1, unevenness 0.3), every rule on, flat clocks (clock_scale 1e9), where the
rules say the fluid's energy is kept while nothing else changes:
  - frozen: after 3 steps the whole fluid is set flowing along x (its phase turned 0.3
    more per box), then 4 runs of energy_flow alone at courant 0.0125; the clocked
    fluid energy falls by the heat booked, up to the flow's own splitting error: the
    gap is at most twice what the same start loses with friction off, plus 1% of the
    heat (friction now rubs only flow crossing a seam, where the flow is pinched, so
    the heat, about 5, is no longer large beside that error, about 0.1 to 0.6);
  - stepped: 8 whole steps at courant 0.0125; every step books heat and the balance
    moves by at most 2e-4 of the clocked fluid energy (the splitting floor: measured
    1.8 of 13,000 in the first step, right after the start's size correction, then
    under 0.13; the same with the pull off, so the pull's books, V n / 2 with the held
    field's half booked as pull work in step.py, add nothing).
"""
import numpy as np

from physics.potts import Settings, cell_sums, energy_flow, fluid_energy, new_universe, stats, step
from physics.potts.region_host import region_host
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_frozen_fluid_loses_exactly_the_heat_friction_books():
    settings = Settings.all_on(clock_scale=1e9, courant=0.0125)
    universe = new_universe(4, 4, seed=1, unevenness=0.3)
    for _ in range(3):
        step(universe, settings)
    psi = universe.psi.numpy()
    turn = 0.3 * np.arange(universe.side)[:, None, None]
    flowing = np.stack([psi[..., 0] * np.cos(turn) - psi[..., 1] * np.sin(turn),
                        psi[..., 0] * np.sin(turn) + psi[..., 1] * np.cos(turn)], axis=-1)
    losses = []
    for rule in (settings, Settings.all_on(clock_scale=1e9, courant=0.0125, friction=0.0)):
        universe.psi.assign(flowing)
        cell_sums(universe, rule)
        start = fluid_energy(universe, rule, clocked=True)
        heat_start = float(region_host(universe, universe.heat).sum())
        for _ in range(4):
            energy_flow(universe, rule)
        losses.append((start - fluid_energy(universe, rule, clocked=True),
                       float(region_host(universe, universe.heat).sum()) - heat_start))
    (lost, heat), (floor, _) = losses
    print(f"fluid energy lost {lost:.4f}, heat booked {heat:.4f}, lost with friction off {floor:.4f}")
    assert heat > 0
    assert abs(lost - heat) <= 2.0 * abs(floor) + 0.01 * heat


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
        assert abs(moved) <= 2e-4 * after["clocked_fluid_energy"]
        assert abs(after["relative_drift"]) < 1e-12
        before = after
