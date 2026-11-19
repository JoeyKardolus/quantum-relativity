"""Calm to own and back loses nothing: whole quanta, the same swirl, and no other cell is touched."""
import numpy as np

from physics.engine import calm_cells, one_cell, step
from physics.engine.axiom_3_structure.definitions.store import peek
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of, swirl_of
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.engine.axiom_3_structure.logic.take_own import take_own


def test_take_own_then_give_back_at_once_is_exact_box_for_box():
    store = one_cell()
    own = store.box_quanta.numpy()[0].copy()
    assert int(own.sum()) == 648_000 == peek(store.cell_quanta, 0)
    assert give_back(store, 0) and store.count == 0 and store.ledger == 0
    quanta, _, slot = boxes_of(store, 0)          # calm: the rule's boxes
    assert np.array_equal(quanta.numpy()[slot], own) and peek(store.cell_quanta, 0) == 648_000


def test_taking_a_second_cell_own_leaves_the_first_one_s_boxes():
    store = calm_cells(2, own_cap=2)
    take_own(store, 1)
    first = store.box_quanta.numpy()[0].copy(), store.box_phase.numpy()[0].copy()
    take_own(store, 0)
    assert np.array_equal(store.box_quanta.numpy()[0], first[0])
    assert np.array_equal(store.box_phase.numpy()[0], first[1])
    assert np.array_equal(store.box_quanta.numpy()[1], first[0])


def test_after_one_round_the_cell_goes_back_and_the_calm_swirl_has_turned_with_it():
    store = one_cell()
    for _ in range(25):                            # one round of the whirl's phase at 6 across
        step(store)
    stepped, stepped_phase = store.box_quanta.numpy()[0].astype(int), store.box_phase.numpy()[0].copy()
    assert give_back(store, 0) and peek(store.cell_quanta, 0) == 648_000
    assert store.ledger == np.abs(stepped - boxes_of(store, 0)[0].numpy()[0]).sum() // 2 > 0
    take_own(store, 0)
    assert int(store.box_quanta.numpy()[0].sum()) == 648_000
    apart = np.angle(np.exp(1j * (store.box_phase.numpy()[0] - stepped_phase)))
    print(f"largest phase difference between the rule and the stepped fluid: {np.abs(apart).max():.4f} rad")
    assert np.abs(apart).max() < 0.01              # the rule's phase is where the stepped fluid's is (0.0012)


def test_a_cell_left_alone_stays_within_counting_noise_of_its_rule_and_can_go_back():
    store, most = one_cell(), 0.0
    while store.step < 6000:
        step(store)
        if store.step % 500 == 0:
            most = max(most, swirl_of(store, 0)["off_rule"])
    print(f"off_rule at most {most:.2f} in 6,000 steps (6 across, 3,000 quanta a box)")
    assert most < 1.0 and give_back(store, 0) and store.count == 0  # measured: 0.38


def test_a_cell_knocked_far_from_the_rule_stays_own():
    store = one_cell()
    quanta = store.box_quanta.numpy()
    quanta[0, :6], quanta[0, 6:12] = quanta[0, :6] + 1000, quanta[0, 6:12] - 1000  # books kept
    store.box_quanta.assign(quanta)
    assert swirl_of(store, 0)["off_rule"] > 1.0 and not give_back(store, 0) and store.count == 1
