"""The squeeze is read off the cells' quanta, and it presses the joint's facing fluid, unevenly."""
import numpy as np

from physics.engine import step, three_cells
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_5_packing.definitions.pull_push import GIVE, PULL
from physics.engine.axiom_5_packing.logic.press import press
from physics.engine.axiom_5_packing.logic.squeeze import pull_push

EVEN = (3 + np.pi / 6) ** (2 / 3)  # two calm volumes' distance squared across a fresh joint


def test_the_squeeze_is_the_pull_across_and_the_push_back_of_the_cells_as_they_are():
    uneven = (3 + np.pi / 12 * (1.5 ** -3 + 0.5 ** -3)) ** (2 / 3)  # the distance squared
    assert np.allclose(pull_push(three_cells(), 1), [PULL * 0.75 / uneven, 1])
    assert np.allclose(pull_push(three_cells(parts=(2, -2, 2)), 1), [PULL / EVEN, 1])
    assert np.allclose(pull_push(three_cells(parts=(3, 0, 1)), 1), [0, 0])  # no joint between


def test_one_press_turns_only_the_joint_s_facing_phases_by_what_it_gives():
    store = three_cells(parts=(2, -2, 2))
    quanta, before = store.box_quanta.numpy().reshape(3, 6, 6, 6), store.box_phase.numpy().copy()
    pull_push(store, 1)
    press(store)
    turned = (store.box_phase.numpy() - before + np.pi) % (2 * np.pi) - np.pi
    pressed = (quanta[0][:, 5, :] + quanta[2][:, 0, :]) / 6000  # each over its volume's mean box
    gives = GIVE * store.turn_per_step * (PULL / EVEN - 1) / (PULL / EVEN)  # of a whole turn
    joint = turned[1].reshape(6, 6, 6)
    assert np.allclose(joint[:, 0, :], gives * pressed, atol=1e-5) and pressed.std() > 0.1
    assert np.allclose(joint[:, 5, :], gives * pressed, atol=1e-5)
    assert np.abs(joint[:, 1:5, :]).max() == 0 and np.abs(turned[[0, 2]]).max() == 0
    assert (store.box_quanta.numpy().reshape(3, 6, 6, 6) == quanta).all()


def test_a_squeezed_joint_s_swirl_goes_irregular_and_one_not_squeezed_stays_as_laid():
    squeezed, held = three_cells(), three_cells(parts=(2, -6, 2))  # 3 calm: push 81, pull 6
    for _ in range(300):
        step(squeezed)
        step(held)
    assert swirl_of(squeezed, 1)["off_rule"] > 100  # measured: 259
    assert swirl_of(held, 1)["off_rule"] < 1 and swirl_of(held, 1)["turns"] == -1
