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
    for _ in range(92):                            # one round of the swirl at 6 across
        step(store)
    stepped, stepped_phase = store.box_quanta.numpy()[0].astype(int), store.box_phase.numpy()[0].copy()
    assert give_back(store, 0) and peek(store.cell_quanta, 0) == 648_000
    assert store.ledger == np.abs(stepped - boxes_of(store, 0)[0].numpy()[0]).sum() // 2 > 0
    take_own(store, 0)
    assert int(store.box_quanta.numpy()[0].sum()) == 648_000
    apart = np.angle(np.exp(1j * (store.box_phase.numpy()[0] - stepped_phase)))
    print(f"largest phase difference between the rule and the stepped fluid: {np.abs(apart).max():.4f} rad")
    assert np.abs(apart).max() < 0.1               # the rule's phase is where the stepped fluid's is


def test_a_cell_that_walked_too_far_from_the_rule_stays_own():
    store, passed = one_cell(), None
    while store.step < 6000 and passed is None:
        step(store)
        if store.step % 100 == 0 and swirl_of(store, 0)["off_rule"] >= 1.0:
            passed = store.step
    print(f"off_rule passes 1 at step {passed} (seed 1, 6 across, 3,000 quanta a box)")
    assert passed is None or (not give_back(store, 0) and store.count == 1)
