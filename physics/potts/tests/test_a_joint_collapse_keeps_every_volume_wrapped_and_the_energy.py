"""Joint collapse: a joint squeezed past its limit vanishes, the energy volumes round it fall together, wrapped.

In an even start (7**3 cells of 4**3 boxes) the face joint (2, 1, 1), between the
energy volumes (1, 1, 1) and (3, 1, 1), and the corner joint (2, 2, 2), which touches
eight energy volumes only at their corners, each keep a thousandth of their energy (the
rest moved into a volume), so they want a billion times their boxes: squeezed far past
the collapse limit. The collapse is run directly, after the cell sums: the face joint
is gone and its two volumes are one; the corner joint waits (its volumes would hang
together only at corners); no energy volume touches another or empty space (26
neighbours), every cell is one face-connected piece, and the total energy holds to
rounding; 5 steps later the wrapping and the energy still hold.
"""
import numpy as np

from physics.potts import Settings, cell_sums, joint_collapse, stats, step
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import cell_id, connected_everywhere, even_universe, move_energy, wrapped_everywhere


@needs_gpu
def test_a_collapsed_joint_leaves_one_wrapped_energy_volume():
    universe = even_universe(7, 4)
    joint = cell_id(7, 2, 1, 1)
    corner = cell_id(7, 2, 2, 2)
    volume = cell_id(7, 5, 5, 5)
    move_energy(universe, joint, volume, 0.001)
    move_energy(universe, corner, volume, 0.001)
    settings = Settings.all_on()
    start = stats(universe, settings)
    cell_sums(universe, settings)
    assert joint_collapse(universe, settings) == 1
    end = stats(universe, settings)
    owner = universe.owner.numpy()
    alive = universe.alive.numpy()
    assert end["collapses"] == 1
    assert alive[joint] == 0 and alive[corner] == 1
    assert end["cells"] == start["cells"] - 1
    assert end["wrap_breaks"] == 0
    assert wrapped_everywhere(owner, universe.kind.numpy())
    assert abs(end["energy_drift"]) < 1e-9 * start["total_energy"]
    assert not np.any(owner == joint)
    assert connected_everywhere(owner, np.flatnonzero(alive == 1))
    for _ in range(5):
        step(universe, settings)
    later = stats(universe, settings)
    assert later["wrap_breaks"] == 0
    assert abs(later["relative_drift"]) < 1e-10
