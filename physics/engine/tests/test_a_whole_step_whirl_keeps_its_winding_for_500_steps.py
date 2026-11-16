"""A whirl turns only in whole steps: one placed in plain fluid keeps winding exactly 1 for 500 steps.

The fluid's complex number per box is sqrt(|quanta|) e^{i phase}; its flow is how the
phase turns, so the winding round a closed loop of boxes (phase steps wrapped to
(-pi, pi], summed, over 2 pi) is a whole number. Plain fluid: a 40-box cube all plus (one
energy volume, no joints, no empty boxes, no pull: pull_strength 0), +1000 quanta per box
dipping as tanh(r / healing)**2 (at least 1 quantum) towards a whirl line along z
through the middle, phase the angle round that line; healing 1 / sqrt(2 g n) from the
default volume stiffness. `step` runs 500 steps with every other rule at its default
(local clocks on). Every 50 steps the winding round squares of half width 3, 5 and 7
boxes round the line, in every z plane 8..31, is exactly 1, round a square beside the
line exactly 0, and the amount of quanta is identical.
"""
import numpy as np

from physics.engine import Settings, State, step, totals
from physics.engine.tests._gpu import needs_gpu

SIDE = 40
CENTRE = 19.5


def _winding(field: np.ndarray, plane: int, low: tuple, high: tuple) -> float:
    """The phase turned round the square from low to high (x, y) in z plane `plane`, over 2 pi."""
    (x0, y0), (x1, y1) = low, high
    path = ([(x, y0) for x in range(x0, x1)] + [(x1, y) for y in range(y0, y1)]
            + [(x, y1) for x in range(x1, x0, -1)] + [(x0, y) for y in range(y1, y0, -1)])
    values = np.array([field[x, y, plane] for x, y in path])
    return float(np.angle(np.roll(values, -1) * np.conj(values)).sum() / (2.0 * np.pi))


@needs_gpu
def test_a_whirl_keeps_one_whole_turn_and_the_amount_holds():
    settings = Settings(pull_strength=0.0)
    axis = np.arange(SIDE) - CENTRE
    across_x, across_y = np.meshgrid(axis, axis, indexing="ij")
    healing = 1.0 / np.sqrt(2.0 * settings.volume_stiffness)
    density = 1000.0 * np.tanh(np.hypot(across_x, across_y) / healing) ** 2
    quanta = np.maximum(np.rint(density), 1).astype(np.int32)[..., None] * np.ones(SIDE, dtype=np.int32)
    phase = np.arctan2(across_y, across_x)[..., None] * np.ones(SIDE)
    state = State.from_host(quanta, phase)
    start = totals(state)["amount"]
    squares = [((20 - half, 20 - half), (19 + half, 19 + half)) for half in (3, 5, 7)]
    beside = ((4, 4), (10, 10))
    for count in range(500):
        step(state, settings)
        if count % 50 == 49:
            amounts = np.abs(state.quanta.numpy()).astype(np.float64)
            field = np.sqrt(amounts) * np.exp(1j * state.phase.numpy().astype(np.float64))
            found = [_winding(field, plane, low, high) for plane in range(8, 32) for low, high in squares]
            outside = [_winding(field, plane, *beside) for plane in range(8, 32)]
            amount = totals(state)["amount"]
            print(f"step {count + 1}: winding {min(found):.6f} .. {max(found):.6f}, beside "
                  f"{max(abs(value) for value in outside):.1e}, amount {amount} (start {start})")
            assert np.allclose(found, 1.0, atol=1e-6)
            assert np.allclose(outside, 0.0, atol=1e-6)
            assert amount == start
