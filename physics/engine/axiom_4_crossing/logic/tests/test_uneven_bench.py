"""The bench, a volume of 3 parts and one of 1 with a joint between: different amounts alone move nothing; the leak does, slowly; their pull presses them on the joint alike."""
import numpy as np
import pytest

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_2_swirls.logic.turn import turn
from physics.engine.axiom_3_structure.definitions.store import put
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_4_crossing.logic.leak import leak
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP

A, JOINT, B = line_cells()  # the three cells' numbers; their slots are 0, 1 and 2


def _turned(parts: tuple, crossing: bool, sealed: bool = False) -> np.ndarray:
    """A's and B's quanta after a turn of whole whirls that only turn and leak (and cross): no pull is
    ever handed on and nothing moves inside. `sealed`: B is held past the limit, so it leaks nothing back."""
    store = three_cells(parts=parts)
    for _ in range(25):
        if sealed:
            put(store.cell_pull, B, DEEP)
        turn(store)
        if crossing:
            cross(store)
        leak(store)
    return store.cell_quanta.numpy()[[A, B]]


def test_two_neighbours_with_different_amounts_and_no_pull_and_no_motion_exchange_only_the_leak():
    with_crossing, leak_alone = _turned((3, -2, 1), True), _turned((3, -2, 1), False)
    assert np.abs(with_crossing - leak_alone).max() <= 3  # the crossing adds nothing (an odd quantum at most)
    assert leak_alone.sum() == 1_296_000 and 972_000 - leak_alone[0] > 1000  # the fuller loses 2,844 a turn


def test_the_leak_alone_is_the_same_whatever_the_neighbour_holds():
    lost = [972_000 - _turned((3, -2, other), False, sealed=True)[0] for other in (1, 3, 12)]
    assert lost[0] == lost[1] == lost[2] > 3000  # 4,572 quanta toward B in a turn, to the quantum


@pytest.fixture(scope="module")
def uneven():
    """The bench (A 3 to B 1) after 2,000 steps with every rule on, its books checked at every one."""
    store = three_cells()
    for _ in range(2000):
        step(store)
        held = store.box_quanta.numpy().astype(np.int64).sum(axis=1)
        assert held[0] + held[2] == 1_296_000
        assert held[1] == -648_000 - store.cell_made.numpy()[JOINT]  # the joint only makes
        assert (held == store.cell_quanta.numpy()[[A, JOINT, B]]).all() and store.box_quanta.numpy()[[0, 2]].min() >= 0
    return store, held


def test_over_80_turns_the_two_only_press_on_the_joint_the_leak_alone_evens_them_and_all_three_stay_whole_whirls(uneven):
    store, held = uneven
    # each is pulled by what the other holds, equal and opposite, so their flows meet in the joint and only
    # press; the leak alone moves energy: 1.5 and 0.5 calm volumes are 1.284 and 0.716 at 80 turns (1.016 and
    # 0.984 while the thinner one could not send back what the fuller sent)
    assert 972_000 - held[0] == held[2] - 324_000 and 300_000 < held[0] - held[2] < 450_000
    for cell, turns in ((A, -1), (JOINT, 1), (B, -1)):
        assert swirl_of(store, cell)["turns"] == (turns,) * 3
