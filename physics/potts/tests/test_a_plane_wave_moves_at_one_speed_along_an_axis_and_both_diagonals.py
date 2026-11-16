"""The energy fluid has no favourite direction: a moving wave packet keeps the same speed along an axis and both diagonals.

The Planck boxes form a cubic grid, so the hops (hop.py) run only along x, y and z. A
direction-free fluid still carries a wave packet at one speed whichever way it points.
One energy volume fills a cube of 40 boxes a side (no joints, no friction, no pull, no
box moves: only the fluid's own steps, energy_flow). A Gaussian packet (width 4 boxes)
is given a phase rising 0.3 per box along x, along the face diagonal (1, 1, 0) / sqrt 2
and along the body diagonal (1, 1, 1) / sqrt 3. The push is set weak (sound 1e-3, chosen
before the run) so the packet is a linear wave and its known speed is the lattice's
group velocity, sum over axes of u_i sin(0.3 u_i) for direction u. After 24 steps the
packet's energy centre has moved that far along its direction within 5%, the three
speeds agree within 5%, it has drifted sideways less than 0.05 boxes, and the total
energy (sum of |psi|**2) holds to float64 rounding.
"""
import numpy as np

from physics.potts import ENERGY_VOLUME, Settings, Universe, cell_sums, energy_flow
from physics.potts.tests._gpu import needs_gpu

SIDE = 40
WAVE = 0.3
WIDTH = 4.0
STEPS = 24
DIRECTIONS = {"axis": (1.0, 0.0, 0.0), "face_diagonal": (1.0, 1.0, 0.0), "body_diagonal": (1.0, 1.0, 1.0)}


def _run(unit: np.ndarray, settings: Settings) -> tuple:
    """How far the packet's energy centre went along `unit` and across it, its known distance, the energy drift."""
    places = np.stack(np.meshgrid(*[np.arange(SIDE, dtype=np.float64)] * 3, indexing="ij"), axis=-1)
    offsets = places - ((SIDE - 1) / 2.0 - 0.2 * SIDE * unit)
    energy = 5.0 * np.exp(-(offsets ** 2).sum(axis=-1) / (2.0 * WIDTH ** 2))
    owner = np.zeros((SIDE,) * 3, dtype=np.int32)
    universe = Universe.from_host(owner, energy, np.array([ENERGY_VOLUME]), SIDE, seed=1,
                                  phase=offsets @ (WAVE * unit))
    universe.corrected = True
    start = _centre(universe.psi.numpy())
    for _ in range(STEPS):
        cell_sums(universe, settings)
        energy_flow(universe, settings)
    psi = universe.psi.numpy()
    moved = _centre(psi) - start
    along = float(moved @ unit)
    known = float((unit * np.sin(WAVE * unit)).sum()) * STEPS * settings.step_time
    drift = abs(float((psi ** 2).sum()) - universe.start_total) / universe.start_total
    return along, float(np.linalg.norm(moved - along * unit)), known, drift


def _centre(psi: np.ndarray) -> np.ndarray:
    energy = (psi ** 2).sum(axis=-1)
    places = np.stack(np.meshgrid(*[np.arange(SIDE, dtype=np.float64)] * 3, indexing="ij"), axis=-1)
    return (places * energy[..., None]).sum(axis=(0, 1, 2)) / energy.sum()


@needs_gpu
def test_a_packet_goes_as_fast_along_an_axis_as_along_either_diagonal():
    settings = Settings.all_on(sound=1e-3, friction=0.0)
    speeds = {}
    for name, direction in DIRECTIONS.items():
        unit = np.asarray(direction) / np.linalg.norm(direction)
        along, across, known, drift = _run(unit, settings)
        speeds[name] = along / (STEPS * settings.step_time)
        print(f"{name}: along {along:.4f} known {known:.4f} ratio {along / known:.4f} across {across:.2e} "
              f"drift {drift:.1e}")
        assert abs(along / known - 1.0) <= 0.05
        assert across < 0.05
        assert drift < 1e-12
    assert max(speeds.values()) <= 1.05 * min(speeds.values())
