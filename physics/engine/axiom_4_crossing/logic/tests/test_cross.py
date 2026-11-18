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
    store, heat = three_cells(), np.zeros(3)
    for _ in range(2000):
        step(store)
        held = store.box_quanta.numpy().astype(np.int64).sum(axis=1)
        assert held[0] + held[2] == 1_296_000 and held[1] == -648_000
        assert (held == store.cell_quanta.numpy()).all() and store.box_quanta.numpy()[[0, 2]].min() >= 0
        assert (store.cell_heat.numpy() >= heat).all() and store.cell_heat.numpy()[1] == 0
        heat = store.cell_heat.numpy()
    assert abs(held[0] - held[2]) < 0.01 * 1_296_000 and heat[2] > 0  # A 3 to B 1 has evened out


def test_nothing_crosses_where_no_joint_lies_between():
    store = three_cells(parts=(3, 0, 1))
    for _ in range(200):
        step(store)
    assert store.cell_quanta.numpy().tolist() == [972_000, 0, 324_000]
    assert store.cell_heat.numpy().max() == 0


def test_equal_volumes_of_still_fluid_exchange_nothing():
    store = _still((2, -2, 2))
    for _ in range(50):
        step(store)
    assert (store.box_quanta.numpy()[[0, 2]] == 3000).all() and store.cell_heat.numpy().max() == 0


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


def test_friction_books_its_stated_share_of_what_crosses():
    store = _still((3, -2, 1))
    joint = -store.box_quanta.numpy()[1].reshape(6, 6, 6).mean(axis=1).ravel()  # minus, a row's box
    share = joint / (joint + 3000)  # the two facing boxes hold 4,500 and 1,500: 3,000 a box
    cross(store)
    crossed = (4500 - store.box_quanta.numpy()[0].reshape(6, 6, 6)[:, 5, :]).ravel()
    assert crossed.min() > 0 and crossed.sum() == store.cell_quanta.numpy()[2] - 324_000
    assert abs(store.cell_heat.numpy()[2] - (share * crossed).sum()) < 12  # 36 whole draws
    assert store.cell_heat.numpy()[0] == 0


def test_the_volumes_keep_their_whole_swirls_while_energy_crosses():
    store = three_cells()
    for _ in range(300):
        step(store)
    left, right = swirl_of(store, 0), swirl_of(store, 2)
    assert (left["turns"], right["turns"]) == (-1, 1)  # B is laid as A's mirror
    assert left["core"] < 0.3 and right["core"] < 0.3
