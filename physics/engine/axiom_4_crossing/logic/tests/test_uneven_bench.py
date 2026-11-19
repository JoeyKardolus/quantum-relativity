"""The bench, a volume of 3 parts and one of 1 with a joint between: exact books, whole whirls, and where it stops."""
import numpy as np
import pytest

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of

A, JOINT, B = line_cells()  # the three cells' numbers; their slots are 0, 1 and 2


@pytest.fixture(scope="module")
def uneven():
    """The bench (A 3 to B 1) after 2,000 steps, its books checked at every one."""
    store = three_cells()
    for _ in range(2000):
        step(store)
        held = store.box_quanta.numpy().astype(np.int64).sum(axis=1)
        assert held[0] + held[2] == 1_296_000
        assert held[1] == -648_000 - store.cell_made.numpy()[JOINT]  # the joint only makes
        assert (held == store.cell_quanta.numpy()[[A, JOINT, B]]).all() and store.box_quanta.numpy()[[0, 2]].min() >= 0
    return store, held


def test_the_books_are_exact_over_2000_steps_and_energy_goes_from_the_fuller_volume(uneven):
    store, held = uneven
    assert 100_000 < 972_000 - held[0] == held[2] - 324_000  # measured: 120,792 crossed from A to B
    for cell, turns in ((A, -1), (JOINT, 1), (B, -1)):  # and all three are whole whirls still
        assert swirl_of(store, cell)["turns"] == (turns,) * 3


@pytest.mark.xfail(strict=True, reason="A 3 to B 1 stops evening out at 851,208 to 444,792 (from 300 "
                   "steps on): the facing ring is left partly standing, so the two caps hold alike")
def test_two_uneven_volumes_even_out(uneven):
    assert abs(uneven[1][0] - uneven[1][2]) < 0.01 * 1_296_000
