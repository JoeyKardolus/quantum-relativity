"""Flipping a cell's sign reverses its turn and nothing else."""
import numpy as np

from physics.engine import one_cell, step
from physics.engine.axiom_2_swirls.logic.flip import flip
from physics.engine.axiom_3_structure.definitions.size_law import size_of
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of


def _mirror(boxes: np.ndarray) -> np.ndarray:
    """A block's numbers with the boxes round each core in the opposite order."""
    return boxes.reshape(6, 6, 6)[:, ::-1, :].ravel()


def test_a_flip_reverses_the_turns_and_keeps_every_quantum_the_size_and_the_speed():
    store = one_cell()
    for _ in range(100):
        step(store)
    quanta, phase, seen = store.box_quanta.numpy()[0].copy(), store.box_phase.numpy()[0].copy(), swirl_of(store, 0)
    flip(store, 0)
    flipped = swirl_of(store, 0)
    assert seen["turns"] == (-1, -1, -1) and flipped["turns"] == (1, 1, 1)
    assert np.array_equal(store.box_quanta.numpy()[0], -_mirror(quanta))  # box for mirrored box
    assert np.array_equal(store.box_phase.numpy()[0], _mirror(phase))
    assert np.allclose(flipped["round_flow"], -np.array(seen["round_flow"])) and flipped["still"] == 0.0
    held = int(store.cell_quanta.numpy()[0])
    assert held == -648_000 == int(store.box_quanta.numpy()[0].sum()) and size_of(held, 6) == 1.0


def test_the_flipped_cell_runs_on_turning_the_other_way_as_fast_and_as_whole():
    plain, flipped = one_cell(), one_cell()
    flip(flipped, 0)
    for _ in range(500):
        step(plain)
        step(flipped)
    one, other = swirl_of(plain, 0), swirl_of(flipped, 0)
    assert one["turns"] == (-1, -1, -1) and other["turns"] == (1, 1, 1) and other["still"] == 0.0
    # as much goes round, the other way (within 1%; not box for box: the odd quantum goes by
    # the giving box's phase, and the mirror swaps which box of a pair gives)
    assert np.allclose(other["round_flow"], -np.array(one["round_flow"]), rtol=0.01)
    assert other["off_rule"] < 1.0 and int(flipped.box_quanta.numpy()[0].sum()) == -648_000
    assert np.abs(flipped.box_phase.numpy()[0] - _mirror(plain.box_phase.numpy()[0])).max() < 0.05
    flip(flipped, 0)  # and back: a volume again
    assert swirl_of(flipped, 0)["turns"] == (-1, -1, -1) and int(flipped.cell_quanta.numpy()[0]) == 648_000
