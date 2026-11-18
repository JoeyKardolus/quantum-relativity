"""Four volumes in a square: exact books, crossing only through a joint, squeezed only between two."""
import numpy as np

from physics.engine import step
from physics.engine.axiom_3_structure.definitions.store import put, slot_of
from physics.engine.axiom_3_structure.logic.lay_square import four_volumes
from physics.engine.axiom_5_packing.logic.squeeze import squeeze

VOLUMES, JOINTS = [0, 2, 6, 8], [1, 3, 5, 7]


def _books(store) -> np.ndarray:
    """Every cell's quanta, checked against the sum of its own boxes."""
    held, cells = store.cell_quanta.numpy(), store.own_cell.numpy()[:store.count]
    assert (store.box_quanta.numpy()[:store.count].astype(np.int64).sum(axis=1) == held[cells]).all()
    return held


def test_the_books_are_exact_and_the_joints_are_squeezed_to_the_balance():
    store = four_volumes()
    for _ in range(1500):
        step(store)
        held = _books(store)
        assert held[VOLUMES].sum() == 4 * 648_000  # plus energy only moves
        assert (held[JOINTS] == -648_000 - store.cell_made.numpy()[JOINTS]).all()  # joints only make
    assert np.abs(held[VOLUMES] - 648_000).max() < 0.01 * 648_000  # 3, 1, 1, 3 halves evened out
    squeeze(store)
    pull, push, apart, _axis = store.own_squeeze.numpy()[[slot_of(store, one) for one in JOINTS]].T
    assert np.abs(pull - push).max() < 0.01 * push.min()        # every joint pushes its pull back
    assert np.abs(apart - 1.25).max() < 0.005                   # from 1.73 apart: a quarter thin
    assert np.abs(held[JOINTS] / -648_000 - 1.28).max() < 0.005


def test_energy_crosses_only_through_a_joint_between_two_volumes():
    store = four_volumes()
    for dead in (1, 5, 7):  # only the joint between the first volume and the third is left
        put(store.cell_quanta, dead, 0)
        store.box_quanta[slot_of(store, dead)].zero_()
    for _ in range(300):
        step(store)
    held = _books(store)
    assert held[2] == 324_000 and held[8] == 972_000 and store.cell_heat.numpy()[[2, 8]].max() == 0
    assert held[6] > 324_000 + 100_000 and held[0] + held[6] == 1_296_000  # down the one joint


def test_a_joint_with_a_volume_on_one_side_only_is_not_squeezed():
    store = four_volumes(parts=(3, 0, 1, 3))  # no second volume: joints 1 and 5 lie at an edge
    for _ in range(300):
        step(store)
    held, made = _books(store), store.cell_made.numpy()
    assert held[1] == held[5] == -648_000 and made[1] == made[5] == 0  # fresh as they were laid
    assert made[3] > 10_000 and made[7] > 10_000 and held[0] + held[6] + held[8] == 7 * 324_000
