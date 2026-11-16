"""Axiom 12: joints oscillate and carry swirl.

Bible rule 5 (joints are the springs of the grid: they keep swinging smaller and bigger
round their balance) and audit answers (owner, 2026-11-16: joints carry swirl and
oscillate); bible rule 4 (whirls turn only in whole steps). Measured:
  1. default universe (seed 1, Settings(), axiom_run.py): every joint's size (its
     corner volume, every step for 300 steps) reverses direction at least 3 times, a
     reversal counted only after a swing of at least SWING boxes either way (so
     rounding jitter is not a swing); threshold set before the run
  2. staged: a 40-box cube of joint boxes (-1000 each, one joint, no lattice) with one
     whirl along z through its middle (density dipping as tanh(r / healing)**2 towards
     the line, phase the angle round it, as test_a_whole_step_whirl_keeps_its_winding),
     Settings(): every 50 steps for 300 steps, the winding round squares of half width
     3, 5 and 7 boxes round the line in every z plane 8..31 is exactly 1, and the amount
     is exact
"""
import numpy as np

from physics.engine import Settings, State, step, totals
from physics.engine.tests._gpu import needs_gpu

SWING = 0.5
SIDE = 40
CENTRE = 19.5


def _reversals(series: np.ndarray) -> int:
    """Turns of direction in `series`, each after a swing of at least SWING since the last turn."""
    turns, direction, anchor = 0, 0, series[0]
    for value in series[1:]:
        if direction >= 0 and value <= anchor - SWING:
            turns += direction == 1
            direction, anchor = -1, value
        elif direction <= 0 and value >= anchor + SWING:
            turns += direction == -1
            direction, anchor = 1, value
        elif (direction == 1 and value > anchor) or (direction == -1 and value < anchor):
            anchor = value
    return turns


def _winding(field: np.ndarray, plane: int, low: tuple, high: tuple) -> float:
    """The phase turned round the square from low to high (x, y) in z plane `plane`, over 2 pi."""
    (x0, y0), (x1, y1) = low, high
    path = ([(x, y0) for x in range(x0, x1)] + [(x1, y) for y in range(y0, y1)]
            + [(x, y1) for x in range(x1, x0, -1)] + [(x0, y) for y in range(y1, y0, -1)])
    values = np.array([field[x, y, plane] for x, y in path])
    return float(np.angle(np.roll(values, -1) * np.conj(values)).sum() / (2.0 * np.pi))


@needs_gpu
def test_every_joint_swings_back_and_forth():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    joints = np.flatnonzero(run["kinds"] == -1)
    sizes = np.stack([row["cell_size"][joints] for row in run["steps"]])
    turns = np.array([_reversals(sizes[:, column]) for column in range(joints.size)])
    swing = sizes.max(axis=0) - sizes.min(axis=0)
    print(f"{joints.size} joints over 300 steps: reversals min {turns.min()}, median {int(np.median(turns))}, max "
          f"{turns.max()}; joints with fewer than 3: {int((turns < 3).sum())}; size range min {swing.min():.2f}, "
          f"median {np.median(swing):.2f} boxes")
    assert turns.min() >= 3


@needs_gpu
def test_a_whirl_in_a_joint_keeps_one_whole_turn():
    settings = Settings()
    healing = 1.0 / np.sqrt(2.0 * settings.push * settings.push_steepness)
    axis = np.arange(SIDE) - CENTRE
    across_x, across_y = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(across_x, across_y)
    density = 1000.0 * np.tanh(radius / healing) ** 2
    quanta = -(np.maximum(np.rint(density), 1).astype(np.int32)[..., None] * np.ones(SIDE, dtype=np.int32))
    phase = np.arctan2(across_y, across_x)[..., None] * np.ones(SIDE)
    state = State.from_host(quanta, phase)
    start = totals(state)["amount"]
    squares = [((20 - half, 20 - half), (19 + half, 19 + half)) for half in (3, 5, 7)]
    worst = 0.0
    for count in range(300):
        step(state, settings)
        if count % 50 == 49:
            amounts = np.abs(state.quanta.numpy()).astype(np.float64)
            field = np.sqrt(amounts) * np.exp(1j * state.phase.numpy().astype(np.float64))
            found = [_winding(field, plane, low, high) for plane in range(8, 32) for low, high in squares]
            worst = max(worst, float(np.abs(np.array(found) - 1.0).max()))
            assert totals(state)["amount"] == start
    print(f"whirl in a joint, 300 steps: largest winding miss from 1: {worst:.2e}")
    assert worst < 1e-6
