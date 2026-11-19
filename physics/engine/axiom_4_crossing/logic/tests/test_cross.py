"""Energy crosses a joint in whole quanta with exact books, only through minus, and still moving."""
import numpy as np

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_2_swirls.definitions.ring_boxes import cap_boxes
from physics.engine.axiom_2_swirls.logic.read_swirl import flow_along
from physics.engine.axiom_4_crossing.logic.cross import cross

A, JOINT, B = line_cells()  # the three cells' numbers; their slots are 0, 1 and 2
FRONT, BACK = cap_boxes(6, 1, 1), cap_boxes(6, 1, 0)  # the caps that face on and back along the line


def _still(parts: tuple, a_phase=0.0, ahead: float = 0.0):
    """Three cells whose volumes hold even, still fluid; volume A's phase may be given. B starts
    in step with A's facing cap (on by the joint's own turn between its two caps), A `ahead` of that."""
    store = three_cells(parts=parts)
    quanta, phase = store.box_quanta.numpy(), store.box_phase.numpy()
    quanta[0], quanta[2] = parts[0] * 1500, parts[2] * 1500
    phase[0] = a_phase
    phase[2] = phase[0][FRONT[0]] + phase[1][FRONT[0]] - phase[1][BACK[0]]
    phase[0] += ahead
    store.box_quanta.assign(quanta)
    store.box_phase.assign(np.angle(np.exp(1j * phase)).astype(np.float32))
    return store


def test_nothing_crosses_where_no_joint_lies_between():
    store = three_cells(parts=(3, 0, 1))
    for _ in range(200):
        step(store)
    assert store.cell_quanta.numpy()[[A, JOINT, B]].tolist() == [972_000, 0, 324_000]


def test_equal_volumes_in_step_exchange_nothing():
    for store in (_still((2, -2, 2)), _still((3, -2, 1)), three_cells(parts=(2, -2, 2))):  # still, uneven, whirls
        start = store.box_quanta.numpy().copy()
        for _ in range(50):
            cross(store)
        assert (store.box_quanta.numpy() == start).all()
    for _ in range(300):
        step(store)  # with all rules on: the two leaks pass each other and the odd quanta drift (measured: 177)
    assert abs(int(store.cell_quanta.numpy()[A]) - 648_000) < 500


AROUND, RING = np.indices((6, 6, 6))[1].ravel(), np.indices((6, 6, 6))[0].ravel() // 2 == 1
FAR = np.minimum((AROUND - 4) % 6, (4 - AROUND) % 6) * RING  # boxes from the back cap, in ring 1


def test_energy_arrives_still_moving_the_way_it_went():
    for way in (1, -1):  # A's fluid moving toward B, then away from it
        store = _still((2, -2, 2), a_phase=-way * 0.3 * FAR)
        sent = flow_along(store.box_quanta.numpy()[0], store.box_phase.numpy()[0], 1)
        for _ in range(30):
            step(store)
        arrived = flow_along(store.box_quanta.numpy()[2], store.box_phase.numpy()[2], 1)
        assert sent * way > 0 and arrived * way > 0.15 * abs(sent)  # measured: 0.17 and 0.82 of it


def test_what_friction_holds_back_stays_in_the_volume_it_came_from():
    crossed = {}
    for joint in (-2, -8):  # A's fluid a step ahead; the fuller joint rubs more: less crosses
        store = _still((3, joint, 1), ahead=0.3)
        cross(store)
        left = 4500 - store.box_quanta.numpy()[0]
        assert left[FRONT].min() > 0 and left.sum() == left[FRONT].sum() == store.cell_quanta.numpy()[B] - 324_000
        assert store.cell_quanta.numpy()[[A, B]].sum() == 1_296_000  # nothing goes anywhere else
        crossed[joint] = left.sum()
    assert crossed[-8] < 0.5 * crossed[-2]  # measured: 144 and 438
