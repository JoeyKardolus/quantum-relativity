"""The leak runs along the seams (rule 6) and friction at A-B faces makes heat, the energy exact.

In an even start every energy-volume box is given a flow along +x (its phase turning
0.5 per box), joints at rest. After one set of hops across the x faces the energy
volumes have sent energy into joints only at the joints' seam boxes (a joint box with
another joint among its 26 neighbours: where joints meet in the web), never through a
joint's middle; the flow energy lost crossing between energy volume and joint is booked
as heat, only in joint boxes beside an energy volume; and total energy (sum of
|psi|**2) is unchanged to rounding.
"""
import numpy as np

from physics.potts import JOINT, Settings, cell_sums, hop, seam_flags, time_levels
from physics.potts.tests._gpu import needs_gpu
from physics.potts.tests._start import box_energy, even_universe


@needs_gpu
def test_joints_take_energy_in_only_at_seam_boxes_and_book_heat():
    universe = even_universe(7, 4)
    universe.corrected = True
    owner = universe.owner.numpy()
    kind = universe.kind.numpy()
    is_joint = (owner >= 0) & (kind[np.maximum(owner, 0)] == JOINT)
    is_volume = (owner >= 0) & ~is_joint
    energy = box_energy(universe)
    turn = np.where(is_volume, 0.5 * np.arange(universe.side)[:, None, None], 0.0)
    amplitude = np.sqrt(energy)
    universe.psi.assign(np.stack([amplitude * np.cos(turn), amplitude * np.sin(turn)], axis=-1))
    settings = Settings.all_on()
    cell_sums(universe, settings)
    seam_flags(universe)
    time_levels(universe, settings)
    for parity in range(2):
        hop(universe, settings, 0, 0, parity, 1.0)
    after = box_energy(universe)
    seam = universe.seam.numpy() == 1
    gained = is_joint & (after > energy + 1e-12)
    heat = universe.heat.numpy()
    print(f"joint boxes that gained: {gained.sum()}, of them seam boxes: {(gained & seam).sum()}, "
          f"heat {heat.sum():.4f}")
    assert gained.sum() > 0
    assert np.all(seam[gained])
    assert heat.sum() > 0 and np.all(heat[~is_joint] == 0)
    assert abs(after.sum() - energy.sum()) < 1e-10 * energy.sum()
