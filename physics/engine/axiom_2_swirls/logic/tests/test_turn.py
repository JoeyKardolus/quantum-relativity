"""Energy is how fast a swirl turns: twice the quanta, twice the turn a step, half the size."""
import numpy as np
import pytest

from physics.engine import calm_cells, step
from physics.engine.axiom_2_swirls.logic.turn import turn_of
from physics.engine.axiom_3_structure.definitions.size_law import size_of
from physics.engine.axiom_3_structure.definitions.store import put
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_3_structure.logic.take_own import take_own

CALM = 648_000


def _turned(quanta: int, steps: int = 1):
    """How far each box's phase went in `steps` steps of a lone cell that holds `quanta`; the store."""
    store = calm_cells(1, 1)
    put(store.cell_quanta, 0, quanta)
    take_own(store, 0)
    before = store.box_phase.numpy()[0].copy()
    for _ in range(steps):
        step(store)
    return np.angle(np.exp(1j * (store.box_phase.numpy()[0] - before))), store


@pytest.mark.parametrize("times", [0.5, 1, 2, 4])
def test_a_cell_turns_in_step_with_its_quanta_and_is_as_much_smaller(times):
    went, store = _turned(int(times * CALM))
    law = turn_of(int(times * CALM), 6, store.turn_per_step)
    assert law == pytest.approx(times * 0.25196, abs=1e-5)  # the stated law: the calm turn times its quanta
    assert np.abs(went - law).max() < 0.002  # every box, within whole quanta's rounding (measured: 0.0003)
    assert size_of(int(times * CALM), 6) == 1 / times
    assert store.own_turned.numpy()[0] == pytest.approx((law - store.turn_per_step) % (2 * np.pi), abs=1e-6)


def test_twice_the_quanta_turns_twice_as_fast_for_a_whole_turn_and_stays_a_whirl():
    calm, calm_store = _turned(CALM, 12)
    twice, store = _turned(2 * CALM, 12)
    assert np.abs(calm - 12 * 0.25196).max() < 0.01           # under half a turn in 12 steps
    assert np.abs(np.angle(np.exp(1j * (twice - 2 * calm)))).max() < 0.01  # twice as far, box for box
    assert swirl_of(store, 0)["turns"] == (-1, -1, -1) and swirl_of(store, 0)["still"] == 0.0
    assert np.array_equal(store.box_quanta.numpy()[0] // 2, calm_store.box_quanta.numpy()[0]) or \
        np.abs(store.box_quanta.numpy()[0] - 2 * calm_store.box_quanta.numpy()[0]).max() < 60


def test_a_joint_turns_as_fast_as_a_volume_of_its_quanta():
    plus, minus = _turned(2 * CALM)[0], _turned(-2 * CALM)[0]
    assert np.abs(minus - plus.reshape(6, 6, 6)[:, ::-1, :].ravel()).max() < 0.002
