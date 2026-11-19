"""Energy crosses a joint only where its own flow carries it, in whole quanta, and takes its flow along: both booked exactly."""
import numpy as np

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_5_packing.logic.lumps import moving

A, JOINT, B = line_cells()  # the three cells' numbers; their slots are 0, 1 and 2


def _crossed(parts: tuple, speed: float, steps: int = 25):
    """The bench with volume A set going toward B at `speed`, after `steps` steps of the fluid inside and the crossing."""
    store = three_cells(parts=parts)
    moving(store, [A], 1, speed)
    for _ in range(steps):
        flow_inside(store)
        cross(store)
    return store


def test_nothing_crosses_where_no_joint_lies_between():
    store = three_cells(parts=(3, 0, 1))
    moving(store, [A], 1, 1.0)
    for _ in range(200):
        step(store)
    assert store.cell_quanta.numpy()[[A, JOINT, B]].tolist() == [972_000, 0, 324_000]


def test_volumes_that_only_hold_different_amounts_exchange_nothing():
    for parts in ((2, -2, 2), (3, -2, 1)):
        store = _crossed(parts, 0.0)
        assert store.cell_quanta.numpy()[[A, B]].tolist() == [parts[0] * 324_000, parts[2] * 324_000]


def test_energy_goes_the_way_its_flow_points_and_arrives_still_moving():
    for way in (1, -1):  # A's fluid going toward B, then away from it: nothing lies that way
        store = _crossed((2, -2, 2), way * 1.0)
        gained = int(store.cell_quanta.numpy()[B]) - 648_000
        flow = store.cell_flow.numpy()[[A, B], 1]
        assert flow.sum() == way * 648_000  # the flow is booked like the quanta: none made, none lost
        if way == 1:
            assert gained > 50_000 and flow[1] > 0.3 * 648_000  # measured: 116,592 quanta in a turn, 0.36 of the flow
        else:
            assert gained == 0 and flow[1] == 0


def test_what_friction_holds_back_stays_in_the_volume_it_came_from_and_its_flow_goes_on():
    crossed = {}
    for joint in (-2, -8):  # the fuller joint rubs more: less crosses
        store = _crossed((3, joint, 1), 1.0)
        held, boxes = store.cell_quanta.numpy(), store.box_quanta.numpy().astype(np.int64).sum(axis=1)
        assert held[[A, B]].sum() == 1_296_000 and held[JOINT] == joint * 324_000  # nothing goes anywhere else
        assert (boxes == held[[A, JOINT, B]]).all() and store.box_quanta.numpy()[[0, 2]].min() >= 0
        assert store.cell_flow.numpy()[[A, B], 1].sum() == 972_000  # held back, the fluid's flow still goes on
        crossed[joint] = int(held[B]) - 324_000
    assert 0 < crossed[-8] < 0.7 * crossed[-2]  # measured: 125,460 and 199,500 quanta in a turn
