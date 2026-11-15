"""A whirl turns only in whole steps: one placed in a uniform region keeps circulation exactly 1 for 500 steps.

The energy fluid is one complex number per Planck box (Gross-Pitaevskii); its flow is
how the phase turns from box to box, so the circulation round any closed loop of boxes
(the phase steps summed, each wrapped to (-pi, pi], over 2 pi) is a whole number: a
whirl (glossary; spin, owner 2026-11-16, is an unbreakable whirl). One energy volume of
20 x 20 x 16 boxes at energy 1 per box, wrapped in a single joint two boxes thick (one
joint has no seams, so nothing leaves the volume), gets a whirl along z through its
middle: phase = the angle round the line between the four middle boxes, the density
dipping to 0 at the line over the healing length (tanh). Only the energy fluid runs
(energy_flow, with the pull), 500 steps: the circulation round squares of half width
3, 5 and 7 boxes round the line, in every inner z plane, stays exactly 1, round a
square beside the whirl exactly 0, and the total energy (sum of |psi|**2) moves by no
more than float64 rounding.
"""
import numpy as np

from physics.potts import ENERGY_VOLUME, JOINT, Settings, Universe, cell_sums, energy_flow, pull
from physics.potts.tests._gpu import needs_gpu

SIDE = 32
CENTRE = 15.5


def _circulation(psi: np.ndarray, plane: int, low: tuple, high: tuple) -> float:
    """The phase turned round the square from low to high (x, y box indices) in z plane `plane`, over 2 pi."""
    (x0, y0), (x1, y1) = low, high
    path = ([(x, y0) for x in range(x0, x1)] + [(x1, y) for y in range(y0, y1)]
            + [(x, y1) for x in range(x1, x0, -1)] + [(x0, y) for y in range(y1, y0, -1)])
    values = np.array([complex(*psi[x, y, plane]) for x, y in path])
    turns = np.angle(np.roll(values, -1) * np.conj(values))
    return float(turns.sum() / (2.0 * np.pi))


@needs_gpu
def test_a_whirl_keeps_one_whole_turn_and_the_energy_holds():
    settings = Settings.all_on()
    owner = np.full((SIDE,) * 3, -1, dtype=np.int32)
    owner[4:28, 4:28, 6:26] = 1
    owner[6:26, 6:26, 8:24] = 0
    axis = np.arange(SIDE) - CENTRE
    dx, dy = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(dx, dy)[..., None] * np.ones(SIDE)
    healing = 1.0 / np.sqrt(2.0 * settings.sound ** 2)
    energy = np.where(owner >= 0, 1.0, 0.0) * np.where(owner == 0, np.tanh(radius / healing) ** 2, 1.0)
    phase = np.where(owner == 0, np.arctan2(dy, dx)[..., None] * np.ones(SIDE), 0.0)
    universe = Universe.from_host(owner, energy, np.array([ENERGY_VOLUME, JOINT]), 4, seed=1, phase=phase)
    universe.corrected = True
    planes = range(10, 22)
    squares = [((16 - half, 16 - half), (15 + half, 15 + half)) for half in (3, 5, 7)]
    beside = ((7, 7), (12, 12))
    start = universe.start_total
    for count in range(500):
        cell_sums(universe, settings)
        pull(universe, settings)
        cell_sums(universe, settings)
        energy_flow(universe, settings)
        if count % 50 == 49:
            psi = universe.psi.numpy()
            found = [_circulation(psi, plane, low, high) for plane in planes for low, high in squares]
            outside = [_circulation(psi, plane, *beside) for plane in planes]
            total = float((psi ** 2).sum())
            print(f"step {count + 1}: circulation {min(found):.12f} .. {max(found):.12f}, beside "
                  f"{max(abs(value) for value in outside):.1e}, energy drift {(total - start) / start:.2e}")
            assert np.allclose(found, 1.0, atol=1e-9)
            assert np.allclose(outside, 0.0, atol=1e-9)
            assert abs(total - start) <= 1e-10 * start
