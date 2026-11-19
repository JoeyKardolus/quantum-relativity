"""A squeezed joint makes quanta out of itself, booked exactly, until its push answers the pull."""
import numpy as np

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_3_structure.definitions.size_law import size_of
from physics.engine.axiom_5_packing.logic.squeeze import pull_push


CELLS = list(line_cells())
JOINT = CELLS[1]


def _run(parts: tuple, steps: int):
    """The store after `steps` steps, each checked: the joint only gains, and only while squeezed."""
    store, plus = three_cells(parts=parts), (parts[0] + parts[2]) * 324_000
    for _ in range(steps):
        before = int(store.cell_quanta.numpy()[JOINT])
        step(store)
        pull, push = store.own_squeeze.numpy()[1, :2]  # the squeeze as the step's making saw it
        held = store.box_quanta.numpy().astype(np.int64).sum(axis=1)
        gained = before - int(held[1])
        assert gained >= 0 and (gained == 0 or pull > push)  # never falls; rises only squeezed
        assert held[1] == parts[1] * 324_000 - store.cell_made.numpy()[JOINT]  # booked to the quantum
        assert held[0] + held[2] == plus and (held == store.cell_quanta.numpy()[CELLS]).all()
    return store


def test_the_joint_shrinks_under_the_squeeze_and_stops_where_push_equals_pull():
    store = _run((2, -2, 2), 600)  # two calm volumes; an uneven pair (3 to 1) does not squeeze: pull 0.88
    pull, push = pull_push(store, JOINT)
    made, joint = store.cell_made.numpy()[JOINT], int(store.cell_quanta.numpy()[JOINT])
    assert abs(pull - push) < 0.001 * pull  # the balance, within a thousandth (measured: 1.0688)
    assert abs(size_of(joint, 6) - 0.9835) < 0.001 and abs(made - 10_884) < 100
    for _ in range(200):
        step(store)
    assert store.cell_made.numpy()[JOINT] - made < 10  # it has stopped


def test_a_stronger_pull_gives_a_smaller_joint():
    weak, strong = _run((2, -2, 2), 400), _run((4, -2, 4), 400)  # twice the energy in the volumes
    assert strong.cell_quanta.numpy()[JOINT] < 1.3 * weak.cell_quanta.numpy()[JOINT] < 0  # more minus


def test_no_pull_no_change():
    for parts in ((0, -2, 0), (2, -6, 2)):  # no volumes; a joint that already pushes harder
        store = _run(parts, 100)
        assert store.cell_made.numpy().max() == 0
        assert store.cell_quanta.numpy()[JOINT] == parts[1] * 324_000
