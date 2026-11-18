"""The volumes' press turns the phase of the joint's facing boxes, and so disturbs its swirl."""
import numpy as np

from physics.engine import step, three_cells
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_5_packing.logic.press import PRESS, press


def test_one_press_turns_only_the_joint_s_facing_phases_by_the_pull_across():
    store = three_cells(parts=(2, -2, 2))
    quanta, before = store.box_quanta.numpy().reshape(3, 6, 6, 6), store.box_phase.numpy().copy()
    press(store)
    turned = (store.box_phase.numpy() - before + np.pi) % (2 * np.pi) - np.pi
    pull = (quanta[0][:, 5, :] + quanta[2][:, 0, :]) / 6000
    joint = turned[1].reshape(6, 6, 6)
    assert np.allclose(joint[:, 0, :], PRESS * pull, atol=1e-5)
    assert np.allclose(joint[:, 5, :], PRESS * pull, atol=1e-5)
    assert np.abs(joint[:, 1:5, :]).max() == 0 and np.abs(turned[[0, 2]]).max() == 0
    assert (store.box_quanta.numpy().reshape(3, 6, 6, 6) == quanta).all()


def test_the_pressed_joint_keeps_its_quanta_and_its_swirl_goes_irregular():
    store = three_cells()
    for _ in range(300):
        step(store)
    joint = swirl_of(store, 1)
    assert store.box_quanta.numpy()[1].sum() == -648_000 and store.box_quanta.numpy()[1].max() <= 0
    assert joint["off_rule"] > 100  # far from the plain swirl it was laid as (measured: 741)
    assert joint["round_flow"] < 0 and joint["turns"] == 1  # measured: by now it turns the other way
