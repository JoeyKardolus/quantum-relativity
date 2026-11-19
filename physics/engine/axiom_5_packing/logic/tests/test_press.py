"""The squeeze is read off the cells' quanta, and it presses the joint's facing fluid."""
import numpy as np

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_2_swirls.definitions.ring_boxes import cap_boxes
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_5_packing.definitions.pull_push import GIVE, PULL
from physics.engine.axiom_5_packing.logic.press import press
from physics.engine.axiom_5_packing.logic.spread import spread
from physics.engine.axiom_5_packing.logic.squeeze import pull_push

JOINT = line_cells()[1]
EVEN = (0.5 + 1 + 0.5) ** 2  # two calm volumes' distance squared across a fresh joint


def _felt(store) -> np.ndarray:
    """The pull across the joint and its push once the pull has been handed to and fro."""
    for _ in range(40):
        spread(store)
    return pull_push(store, JOINT)


def test_the_squeeze_is_the_pull_felt_at_the_two_volumes_and_the_push_back_of_the_joint():
    uneven = (0.5 / 1.5 + 1 + 0.5 / 0.5) ** 2  # the distance squared: half widths and the joint
    assert np.allclose(pull_push(three_cells(), JOINT), [0, 1])  # no pull has arrived yet
    assert np.allclose(_felt(three_cells()), [PULL * 1.7 * 0.7 / uneven, 1])  # 1.5 and 0.5, and a
    assert np.allclose(_felt(three_cells(parts=(2, -2, 2))), [PULL * 1.44 / EVEN, 1])  # fifth back
    assert np.allclose(_felt(three_cells(parts=(3, 0, 1))), [0, 0])  # no joint between


def test_one_press_turns_only_the_joint_s_facing_phases_by_what_it_gives():
    store = three_cells(parts=(2, -2, 2))
    quanta, before = store.box_quanta.numpy().copy(), store.box_phase.numpy().copy()
    front, back = cap_boxes(6, 1, 1), cap_boxes(6, 1, 0)  # the caps that face on and back
    quanta[0, front[:3]], quanta[0, front[3:]] = 3600, 2400  # A presses its rows unevenly
    store.box_quanta.assign(quanta)
    _felt(store)
    press(store)
    turned = (store.box_phase.numpy() - before + np.pi) % (2 * np.pi) - np.pi
    pressed = (quanta[0, front] + quanta[2, back]) / 6000  # each over its volume's mean box
    gives = GIVE * store.turn_per_step * (1 - EVEN / (1.44 * PULL))  # of a whole turn
    assert np.allclose(turned[1, back], gives * pressed, atol=1e-5) and pressed.std() > 0.09
    assert np.allclose(turned[1, front], gives * pressed, atol=1e-5)
    turned[1, front + back] = 0  # nothing else is turned, and no quantum moves
    assert np.abs(turned).max() == 0 and (store.box_quanta.numpy() == quanta).all()


def test_a_squeezed_joint_s_swirl_goes_irregular_and_one_not_squeezed_stays_as_laid():
    squeezed, held = three_cells(parts=(6, -2, 2)), three_cells(parts=(2, -6, 2))  # push 81, pull 2.4
    for _ in range(300):
        step(squeezed)
        step(held)
    assert swirl_of(squeezed, JOINT)["off_rule"] > 100  # measured: 570
    calm = swirl_of(held, JOINT)  # plain counting noise is 1 (measured: 0.007)
    assert calm["off_rule"] < 1 and calm["turns"] == (1, 1, 1)  # a joint: the mirror's turn
