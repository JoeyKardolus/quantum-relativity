"""Joint collapse: a joint squeezed past its limit vanishes, its energy volumes fall together, ABAB holds.

In an even checkerboard one inner joint keeps a thousandth of its energy (the rest is
moved, exactly, into a neighbouring energy volume's box), so it wants a billion times
its boxes: squeezed far past the collapse limit. The collapse is run directly, after
the cell sums: the joint is gone, its six energy volumes are one, nothing like touches
like, every cell is one piece, and not one quantum was lost; 5 steps later ABAB and
the energy still hold.
"""
import numpy as np

from physics.potts import Settings, cell_sums, contact_pairs, joint_collapse, stats, step
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import cell_id, connected_everywhere, even_universe


@needs_gpu
def test_a_collapsed_joint_leaves_one_energy_volume_and_abab():
    universe = even_universe(6, 4)
    joint = cell_id(6, 2, 2, 3)
    volume = cell_id(6, 2, 2, 2)
    owner = universe.owner.numpy()
    energy = universe.energy.numpy()
    kept = energy[owner == joint] // 1000
    moved = int((energy[owner == joint] - kept).sum())
    energy[owner == joint] = kept
    spread, rest = divmod(moved, int(np.sum(owner == volume)))
    energy[owner == volume] += spread
    energy[tuple(np.argwhere(owner == volume)[0])] += rest
    universe.energy.assign(energy)
    settings = Settings.all_on()
    start = stats(universe, settings)
    cell_sums(universe, settings)
    assert joint_collapse(universe, settings) == 1
    end = stats(universe, settings)
    owner = universe.owner.numpy()
    alive = universe.alive.numpy()
    assert end["collapses"] == 1
    assert alive[joint] == 0
    assert end["cells"] == start["cells"] - 5
    assert end["abab_breaks"] == 0
    assert end["energy_drift"] == 0
    assert not np.any(owner == joint)
    low, high, _ = contact_pairs(owner)
    kinds = universe.kind.numpy()
    assert np.all(kinds[low] != kinds[high])
    assert connected_everywhere(owner, np.flatnonzero(alive == 1))
    for _ in range(5):
        step(universe, settings)
    later = stats(universe, settings)
    assert later["abab_breaks"] == 0
    assert later["energy_drift"] == 0
