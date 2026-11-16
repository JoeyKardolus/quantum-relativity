"""The pull obeys Newton's third law: it is felt by the energy volumes only, and its sum over them is about 0.

Bible: A pulls on nearby A's. The pull potential V (pull.py) is sourced by the energy
volumes' energy; the on-site turn (on_site.py) lets only energy-volume boxes feel it.
A small uneven start (7 cells of 4 boxes a side, seed 1, unevenness 0.3), every rule
on, after 3 steps and a fresh solve: one on-site turn with V and one without, from the
same psi, show which boxes feel V (their phases differ); those must be exactly the
energy volumes' boxes. The net pull on them, -sum of |psi|**2 grad V (central
differences), must be under 0.1% of the pull's size there (sum of |psi|**2 |dV/dx|):
the self-pull of a field the A-energy exerts on itself cancels (measured 2026-11-16:
share 0.0002; felt by every box, as before, it was 0.09).
"""
import numpy as np

from physics.potts import ENERGY_VOLUME, Settings, cell_sums, new_universe, on_site, pull, seam_flags, step, time_levels
from physics.potts.tests._gpu import needs_gpu


def _complex(psi: np.ndarray) -> np.ndarray:
    return psi[..., 0] + 1j * psi[..., 1]


@needs_gpu
def test_only_energy_volumes_feel_the_pull_and_it_sums_to_zero():
    settings = Settings.all_on()
    universe = new_universe(7, 4, seed=1, unevenness=0.3)
    for _ in range(3):
        step(universe, settings)
    cell_sums(universe, settings)
    pull(universe, settings)
    cell_sums(universe, settings)
    time_levels(universe, settings)
    seam_flags(universe)
    start = universe.psi.numpy().copy()
    potential = universe.potential.numpy().copy()
    on_site(universe, settings, 0, 1.0)
    with_pull = _complex(universe.psi.numpy())
    universe.psi.assign(start)
    universe.potential.zero_()
    on_site(universe, settings, 0, 1.0)
    without_pull = _complex(universe.psi.numpy())
    universe.psi.assign(start)
    universe.potential.assign(potential)
    felt = np.abs(np.angle(with_pull * np.conj(without_pull))) > 1e-12
    owner = universe.owner.numpy()
    kind = universe.kind.numpy()
    volume_box = (owner >= 0) & (kind[np.maximum(owner, 0)] == ENERGY_VOLUME)
    density = (start ** 2).sum(axis=-1)
    field = potential.astype(np.float64)
    slopes = [(np.roll(field, -1, axis) - np.roll(field, 1, axis)) / 2.0 for axis in range(3)]
    net = [float(-(density * slope)[volume_box].sum()) for slope in slopes]
    size = float(sum((density * np.abs(slope))[volume_box].sum() for slope in slopes))
    print(f"felt boxes {int(felt.sum())}, energy-volume boxes {int(volume_box.sum())}, net pull {np.round(net, 4)}, "
          f"size {size:.2f}, share {max(abs(value) for value in net) / size:.5f}")
    assert np.array_equal(felt, volume_box & (np.abs(field) > 0))
    assert max(abs(value) for value in net) <= 1e-3 * size
