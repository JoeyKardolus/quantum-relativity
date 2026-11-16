"""Report, no verdict: what a whole-step whirl does when the flow pushes it into a joint.

Two energy volumes (A1: x 5..14, A2: x 17..26; y, z 5..26) face each other across two
joints one box thick each (J1 at x 15, J2 at x 16, so every box of both is a seam box),
all wrapped in a third joint three boxes thick; every box holds 1. A whirl of one turn
along z is placed round the line through (9.5, 15.5) in A1, and the whole fluid flows
along +x (its phase rising 0.15 per box), carrying the whirl towards the joints. Only
the energy fluid runs (cell_sums, pull, energy_flow; no box moves), every other rule on,
120 steps, once with friction (the default) and once without. Every 10 steps it prints,
in the middle z plane, the squares a whirl pierces in the band the whirl travels (y
within 5 of 15.5; the closed walls' corners make whirl pairs of their own), counted only
where all four boxes are one cell's (a square across a cell border says nothing: cells
meet only at seams, and a joint one box thick holds no such square), by the cell that
holds them, with the summed winding per cell, and the heat booked (friction's net). Asserted: only that the
numbers are finite and the total energy (sum of |psi|**2) holds to rounding.
"""
import numpy as np
import pytest

from physics.potts import ENERGY_VOLUME, JOINT, Settings, Universe, cell_sums, energy_flow, pull
from physics.potts.tests._gpu import needs_gpu

SIDE = 32
PLANE = 16
NAMES = {0: "A1", 1: "J1", 2: "J2", 3: "A2", 4: "wrap"}


def _start() -> Universe:
    owner = np.full((SIDE,) * 3, -1, dtype=np.int32)
    owner[2:30, 2:30, 2:30] = 4
    owner[5:15, 5:27, 5:27] = 0
    owner[15, 5:27, 5:27] = 1
    owner[16, 5:27, 5:27] = 2
    owner[17:27, 5:27, 5:27] = 3
    axis = np.arange(SIDE, dtype=np.float64)
    whirl = np.arctan2(axis[None, :] - 15.5, axis[:, None] - 9.5)[..., None] * np.ones(SIDE)
    phase = whirl + 0.15 * axis[:, None, None]
    energy = np.where(owner >= 0, 1.0, 0.0)
    kinds = np.array([ENERGY_VOLUME, JOINT, JOINT, ENERGY_VOLUME, JOINT])
    universe = Universe.from_host(owner, energy, kinds, 6, seed=1, phase=phase)
    universe.corrected = True
    return universe


def _pierced(psi: np.ndarray, owner: np.ndarray) -> dict:
    """Winding per cell of the z-normal squares in the middle plane whose four boxes share one cell."""
    field = psi[..., 0] + 1j * psi[..., 1]
    layer = field[:, :, PLANE]
    cells = owner[:, :, PLANE]
    corners = [layer[:-1, :-1], layer[1:, :-1], layer[1:, 1:], layer[:-1, 1:]]
    turn = sum(np.angle(np.conj(corners[index]) * corners[(index + 1) % 4]) for index in range(4))
    winding = np.rint(turn / (2.0 * np.pi)).astype(int)
    owners = [cells[:-1, :-1], cells[1:, :-1], cells[1:, 1:], cells[:-1, 1:]]
    one_cell = (owners[0] >= 0) & (owners[0] == owners[1]) & (owners[0] == owners[2]) & (owners[0] == owners[3])
    winding[~one_cell] = 0
    found = {}
    for place in np.argwhere(winding != 0):
        if abs(place[1] + 0.5 - 15.5) > 5.0:
            continue
        name = NAMES[int(owners[0][tuple(place)])]
        found.setdefault(name, []).append((float(place[0] + 0.5), float(place[1] + 0.5), int(winding[tuple(place)])))
    return found


@needs_gpu
@pytest.mark.parametrize("friction", [1.0, 0.0])
def test_report_a_whirl_pushed_into_a_joint(friction):
    settings = Settings.all_on(friction=friction)
    universe = _start()
    owner = universe.owner.numpy()
    print(f"\nfriction {friction}")
    for count in range(121):
        if count % 10 == 0:
            psi = universe.psi.numpy()
            found = _pierced(psi, owner)
            heat = float(universe.heat.numpy().sum())
            drift = (float((psi ** 2).sum()) - universe.start_total) / universe.start_total
            summary = {name: (sum(turns for _, _, turns in places), [(x, y) for x, y, _ in places][:4])
                       for name, places in found.items()}
            print(f"t {count}: winding per cell {summary}, heat {heat:.3f}, energy drift {drift:.1e}")
            assert np.isfinite(psi).all() and np.isfinite(heat)
            assert abs(drift) < 1e-12
        if count < 120:
            cell_sums(universe, settings)
            pull(universe, settings)
            cell_sums(universe, settings)
            energy_flow(universe, settings)
