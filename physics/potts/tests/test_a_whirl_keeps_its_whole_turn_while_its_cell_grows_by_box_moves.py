"""Box moves keep a box's flow: a whole-step whirl keeps winding exactly 1 while box moves grow its cell, 500 steps.

A box that changes cell keeps the norm, the phase's continuity and the flow
(box_moves.py, part 4). One energy volume of 18**3 boxes (its whirl along z through
the middle, density dipping to 0 at the line over the healing length) inside one joint
two boxes thick, the joint holding as much energy as the volume, both made with
boxes_per_cell_side 20: each wants 20**3 = 8000 boxes, so the volume grows by box
moves into its joint and the joint into empty space (about 12,900 boxes change cell,
the volume reaches 7,200 boxes). Box moves on, pull and friction off
(attraction 0, friction 0), 500 whole steps: the circulation round squares of half
width 3 and 5 round the line, in every inner z plane, stays exactly 1, and the total
energy moves by no more than float64 rounding. The old move rule (home's phase kept,
the joined box a plain copy) kept it too on this start; on a foam start (7 cells of
8 boxes) neither keeps it: the host's borders swing and the whirl leaves the host
within about 20 steps.
"""
import numpy as np

from physics.potts import ENERGY_VOLUME, JOINT, Settings, Universe, stats, step
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
def test_a_whirl_keeps_one_whole_turn_while_box_moves_grow_its_cell():
    settings = Settings.all_on(attraction=0.0, friction=0.0)
    owner = np.full((SIDE,) * 3, -1, dtype=np.int32)
    owner[5:27, 5:27, 5:27] = 1
    owner[7:25, 7:25, 7:25] = 0
    axis = np.arange(SIDE) - CENTRE
    dx, dy = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(dx, dy)[..., None] * np.ones(SIDE)
    healing = 1.0 / np.sqrt(2.0 * settings.sound ** 2)
    energy = np.where(owner == 0, np.tanh(radius / healing) ** 2, 0.0)
    energy[owner == 1] = energy.sum() / np.sum(owner == 1)
    phase = np.where(owner == 0, np.arctan2(dy, dx)[..., None] * np.ones(SIDE), 0.0)
    universe = Universe.from_host(owner, energy, np.array([ENERGY_VOLUME, JOINT]), 20, seed=1, phase=phase)
    universe.corrected = True
    squares = [((16 - half, 16 - half), (15 + half, 15 + half)) for half in (3, 5)]
    moved = 0
    for count in range(500):
        step(universe, settings)
        moved += universe.last_moves
        if count % 50 == 49:
            psi = universe.psi.numpy()
            found = [_circulation(psi, plane, low, high) for plane in range(11, 21) for low, high in squares]
            drift = stats(universe, settings)["relative_drift"]
            print(f"step {count + 1}: circulation {min(found):.12f} .. {max(found):.12f}, boxes moved {moved}, "
                  f"energy drift {drift:.1e}")
            assert np.allclose(found, 1.0, atol=1e-9)
            assert abs(drift) < 1e-10
    assert moved > 1000
