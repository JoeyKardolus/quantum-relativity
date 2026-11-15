"""The leak runs along the seams (rule 6) and friction makes heat, energy exact.

In an even checkerboard every energy-volume box is given a flow along +x, joints at
rest. After one flow substep the energy volumes have sent energy into joints only at
the joints' seam boxes (a joint box sharing an edge with another joint's box), never
through a joint's middle; momentum lost crossing into the joints is booked as heat in
the receiving joint boxes; and total energy is unchanged to the quantum.
"""
import numpy as np

from physics.potts import JOINT, QUANTUM, Settings, cell_sums, flow, seam_flags, time_levels
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import even_universe


@needs_gpu
def test_joints_take_energy_in_only_at_seam_boxes_and_book_heat():
    universe = even_universe(6, 4)
    universe.corrected = True
    owner = universe.owner.numpy()
    kind = universe.kind.numpy()
    energy = universe.energy.numpy()
    is_joint = (owner >= 0) & (kind[np.maximum(owner, 0)] == JOINT)
    momentum = np.zeros(owner.shape + (3,), dtype=np.float32)
    momentum[(owner >= 0) & ~is_joint, 0] = 0.5 * (energy[(owner >= 0) & ~is_joint] * QUANTUM)
    universe.momentum.assign(momentum)
    settings = Settings.all_on()
    cell_sums(universe, settings)
    seam_flags(universe)
    time_levels(universe, settings)
    flow(universe, settings, 0)
    after = universe.energy.numpy()
    seam = universe.seam.numpy() == 1
    gained = is_joint & (after > energy)
    heat = universe.heat.numpy()
    print(f"joint boxes that gained: {gained.sum()}, of them seam boxes: {(gained & seam).sum()}, "
          f"heat {heat.sum():.4f}")
    assert gained.sum() > 0
    assert np.all(seam[gained])
    assert heat.sum() > 0 and np.all(heat[~gained] == 0)
    assert int(after.sum()) == int(energy.sum())
