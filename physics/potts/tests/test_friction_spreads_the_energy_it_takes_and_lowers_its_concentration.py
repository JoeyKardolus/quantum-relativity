"""Friction's lost order spreads energy (rule 8, drift from imperfect transfer): what it touches gets less concentrated.

An even start (7 cells of 4 boxes a side) made uneven in its energy volumes (every box
of each energy volume scaled by a seeded factor between 0.5 and 2), every energy-volume
box given a flow along +x (phase turning 0.5 per box), joints at rest. One set of hops
across the x faces with friction (hop.py) plans the spreading; spread_heat applies it.
Against the same hops whose plan is thrown away: the boxes the spreading touched hold
energy closer to their neighbourhood's mean, so their spread of energies (variance)
falls; the energy spread equals the heat friction booked; some energy crosses from the
energy volumes into the joints; total energy is exact.
"""
import numpy as np
import warp as wp

from physics.potts import ENERGY_VOLUME, Settings, cell_sums, hop, seam_flags, spread_heat, time_levels
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import box_energy, even_universe


def _flowing_start():
    universe = even_universe(7, 4)
    universe.corrected = True
    owner = universe.owner.numpy()
    kind = universe.kind.numpy()
    is_volume = (owner >= 0) & (kind[np.maximum(owner, 0)] == ENERGY_VOLUME)
    factor = np.random.default_rng(5).uniform(0.5, 2.0, size=kind.size)
    energy = box_energy(universe) * np.where(is_volume, factor[np.maximum(owner, 0)], 1.0)
    turn = np.where(is_volume, 0.5 * np.arange(universe.side)[:, None, None], 0.0)
    amplitude = np.sqrt(energy)
    universe.psi.assign(np.stack([amplitude * np.cos(turn), amplitude * np.sin(turn)], axis=-1))
    settings = Settings.all_on()
    cell_sums(universe, settings)
    seam_flags(universe)
    time_levels(universe, settings)
    for parity in range(2):
        hop(universe, settings, 0, 0, parity, 1.0)
    return universe, settings, is_volume


@needs_gpu
def test_friction_spreads_its_heat_and_the_boxes_it_touches_get_less_concentrated():
    universe, settings, is_volume = _flowing_start()
    touched = np.any(universe.spread.numpy() != 0.0, axis=-1)
    before = box_energy(universe)
    spread_heat(universe, settings)
    after = box_energy(universe)
    heat = float(universe.heat.numpy().sum())
    moved = float(universe.spread_moved.numpy()[0])
    to_joints = float(after[~is_volume].sum() - before[~is_volume].sum())
    wp.synchronize()
    print(f"touched boxes {touched.sum()}, variance {before[touched].var():.5f} -> {after[touched].var():.5f}, "
          f"heat {heat:.4f}, spread {moved:.4f}, into joints {to_joints:.4f}")
    assert touched.sum() > 0
    assert after[touched].var() < before[touched].var()
    assert abs(moved - heat) <= 1e-9 * max(heat, 1.0) or moved < heat
    assert moved > 0.5 * heat
    assert to_joints > 0
    assert np.all(universe.spread.numpy() == 0.0)
    assert abs(after.sum() - before.sum()) < 1e-12 * before.sum()
