"""Energy crosses a joint in whole quanta with exact books, only through minus, and still moving."""
import numpy as np

from physics.engine import step, three_cells
from physics.engine.axiom_2_swirls.logic.read_swirl import flow_along
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_4_crossing.logic.cross import cross


def _still(parts: tuple, a_phase=0.0):
    """Three cells whose volumes hold even, still fluid; volume A's phase may be given."""
    store = three_cells(parts=parts)
    quanta, phase = store.box_quanta.numpy(), store.box_phase.numpy()
    quanta[0], quanta[2] = parts[0] * 1500, parts[2] * 1500
    phase[0], phase[2] = a_phase, np.ravel(a_phase)[-1]  # B starts in step with A's facing side
    store.box_quanta.assign(quanta)
    store.box_phase.assign(phase)
    return store


def test_the_books_are_exact_over_2000_steps():
    store = three_cells()
    for _ in range(2000):
        step(store)
        held = store.box_quanta.numpy().astype(np.int64).sum(axis=1)
        assert held[0] + held[2] == 1_296_000
        assert held[1] == -648_000 - store.cell_made.numpy()[1]  # the joint only makes
        assert (held == store.cell_quanta.numpy()[:3]).all() and store.box_quanta.numpy()[[0, 2]].min() >= 0
    assert abs(held[0] - held[2]) < 0.01 * 1_296_000  # A 3 to B 1 has evened out


def test_nothing_crosses_where_no_joint_lies_between():
    store = three_cells(parts=(3, 0, 1))
    for _ in range(200):
        step(store)
    assert store.cell_quanta.numpy().tolist() == [972_000, 0, 324_000]


def test_equal_volumes_of_still_fluid_exchange_nothing():
    store = _still((2, -2, 2))
    for _ in range(50):
        step(store)
    assert (store.box_quanta.numpy()[[0, 2]] == 3000).all()


def test_energy_arrives_still_moving_the_way_it_went():
    line = np.indices((6, 6, 6))[1].ravel()  # a box's place along the line of cells
    arrived = {}
    for way in (1, -1):  # A's fluid moving toward B, then away from it
        store = _still((2, -2, 2), a_phase=-way * 0.3 * line)
        sent = flow_along(store.box_quanta.numpy()[0], store.box_phase.numpy()[0], 1)
        for _ in range(30):
            step(store)
        arrived[way] = flow_along(store.box_quanta.numpy()[2], store.box_phase.numpy()[2], 1)
        assert arrived[way] * way > 0.25 * abs(sent)  # B's fluid moves the way A's went
    assert arrived[1] > 0.5 * sent  # more arrived than friction's half took (measured: 0.69)


def test_what_friction_holds_back_stays_in_the_volume_it_came_from():
    crossed = {}
    for joint in (-2, -8):  # the fuller joint rubs more: less crosses, the rest stays in A
        store = _still((3, joint, 1))
        cross(store)
        left = (4500 - store.box_quanta.numpy()[0].reshape(6, 6, 6)[:, 5, :]).ravel()
        assert left.min() > 0 and left.sum() == store.cell_quanta.numpy()[2] - 324_000
        assert store.cell_quanta.numpy()[[0, 2]].sum() == 1_296_000  # nothing goes anywhere else
        crossed[joint] = left.sum()
    assert crossed[-8] < 0.5 * crossed[-2]


def test_the_volumes_keep_their_whole_swirls_while_energy_crosses():
    store = three_cells()
    for _ in range(300):
        step(store)
    left, right = swirl_of(store, 0), swirl_of(store, 2)
    assert left["turns"] == (-1, -1, -1) and right["turns"] == (1, -1, 1)  # B: A half round the line
    assert left["core"] < 0.3 and right["core"] < 0.3
