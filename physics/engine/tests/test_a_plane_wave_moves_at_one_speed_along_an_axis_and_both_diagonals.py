"""The fluid has no favourite direction: a wave packet of whole quanta keeps one speed along an axis and both diagonals.

The hops (hop.py) run only along x, y and z, and move whole quanta. A direction-free
fluid still carries a packet at one speed whichever way it points. Plain fluid fills a
40-box cube (all plus: no joints, no empty boxes, no pull). A Gaussian packet (width 4
boxes, 100000 quanta at its peak, at least 1 quantum everywhere) gets a phase rising
0.3 per box along x, along the face diagonal (1, 1, 0) / sqrt 2 and along the body
diagonal (1, 1, 1) / sqrt 3. Settings chosen before the run: volume stiffness 1e-6 (a
linear wave, whose known speed is the lattice group velocity, sum over axes of
u_i sin(0.3 u_i) for direction u) and local clocks off (clock_packing 0; on, the packed
peak would run slower than its tail, a separate rule). After 48 steps (12 time units) the
packet's quanta centre has moved that far along its direction within 5%, the three
speeds agree within 5%, it drifts sideways less than 0.05 boxes, and the amount of
quanta is identical.
"""
import numpy as np

from physics.engine import Settings, State, step, totals
from physics.engine.tests._gpu import needs_gpu

SIDE = 40
WAVE = 0.3
WIDTH = 4.0
STEPS = 48
DIRECTIONS = {"axis": (1.0, 0.0, 0.0), "face_diagonal": (1.0, 1.0, 0.0), "body_diagonal": (1.0, 1.0, 1.0)}
PLACES = np.stack(np.meshgrid(*[np.arange(SIDE, dtype=np.float64)] * 3, indexing="ij"), axis=-1)


def _centre(quanta: np.ndarray) -> np.ndarray:
    weight = np.abs(quanta).astype(np.float64)
    return (PLACES * weight[..., None]).sum(axis=(0, 1, 2)) / weight.sum()


def _run(unit: np.ndarray, settings: Settings) -> tuple:
    """How far the packet's quanta centre went along `unit` and across it, its known distance, the amount change."""
    offsets = PLACES - ((SIDE - 1) / 2.0 - 0.15 * SIDE * unit)
    density = 100000.0 * np.exp(-(offsets ** 2).sum(axis=-1) / (2.0 * WIDTH ** 2))
    quanta = np.maximum(np.rint(density), 1).astype(np.int32)
    state = State.from_host(quanta, offsets @ (WAVE * unit))
    start = _centre(quanta)
    amount = totals(state)["amount"]
    for _ in range(STEPS):
        step(state, settings)
    moved = _centre(state.quanta.numpy()) - start
    along = float(moved @ unit)
    known = float((unit * np.sin(WAVE * unit)).sum()) * STEPS * settings.step_time
    return along, float(np.linalg.norm(moved - along * unit)), known, totals(state)["amount"] - amount


@needs_gpu
def test_a_packet_goes_as_fast_along_an_axis_as_along_either_diagonal():
    settings = Settings(pull_strength=0.0, volume_stiffness=1e-6, clock_packing=0.0)
    speeds = {}
    for name, direction in DIRECTIONS.items():
        unit = np.asarray(direction) / np.linalg.norm(direction)
        along, across, known, change = _run(unit, settings)
        speeds[name] = along / (STEPS * settings.step_time)
        print(f"{name}: along {along:.4f} known {known:.4f} ratio {along / known:.4f} across {across:.2e} "
              f"amount change {change}")
        assert abs(along / known - 1.0) <= 0.05
        assert across < 0.05
        assert change == 0
    assert max(speeds.values()) <= 1.05 * min(speeds.values())
