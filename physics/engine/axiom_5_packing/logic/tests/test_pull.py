"""The pull, pair by pair: each volume toward the other by what the other holds, late by the distance, never past the top speed."""
import numpy as np

from physics.engine import line_cells, three_cells
from physics.engine.axiom_5_packing.logic.pull import pull
from physics.engine.axiom_6_limits.definitions.collapse_limit import TOP_SPEED

A, _JOINT, B = line_cells()
APART = 0.5 / 1.5 + 1 + 0.5 / 0.5  # A holds 1.5 calm volumes, B a half: two half widths and the joint


def _pulled(store, steps: int) -> np.ndarray:
    """A's and B's flow along the line after `steps` steps of the pull alone."""
    for _ in range(steps):
        pull(store)
        store.step += 1
    return store.cell_flow.numpy()[[A, B], 1]


def test_before_the_pull_has_had_a_step_to_arrive_there_is_none():
    store = three_cells()
    assert (_pulled(store, 1) == 0).all()
    assert np.allclose(store.cell_pull.numpy()[[A, B]], [1.5, 0.5])  # each lies as deep as its own energy


def test_each_is_pulled_toward_the_other_by_what_the_other_holds_over_the_distance_squared():
    store = three_cells()
    flow = _pulled(store, 2)  # the second step reads what the other held a step ago
    speed = flow / [972_000, 324_000]
    assert np.allclose(speed, np.array([0.5, -1.5]) / APART ** 2 * store.turn_per_step / (2 * np.pi), rtol=1e-3)
    assert abs(int(flow.sum())) <= 1  # equal and opposite: what A gains toward B, B gains toward A
    assert np.allclose(store.cell_pull.numpy()[[A, B]], [1.5 + 0.5 / APART, 0.5 + 1.5 / APART])


def test_a_volume_s_own_energy_does_not_move_it():
    gained = [_pulled(three_cells(parts=(own, -2, 1)), 5)[0] / (own * 324_000) for own in (2, 200)]
    far = [0.5 / (own / 2) + 1 + 1 for own in (2, 200)]  # only its width differs, so the distance does
    assert np.allclose(np.array(gained) * np.array(far) ** 2, gained[0] * far[0] ** 2, rtol=1e-3)


def test_nothing_is_set_going_faster_than_the_top_speed():
    store = three_cells(parts=(2, -2000, 40_000))  # 20,000 calm volumes a thin joint away
    assert _pulled(store, 3)[0] == int(648_000 * TOP_SPEED)
