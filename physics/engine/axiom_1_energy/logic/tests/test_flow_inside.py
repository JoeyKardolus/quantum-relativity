"""The fluid moves in whole quanta: the total is exact, no box goes below 0, and it does move."""
import numpy as np

from physics.engine import calm_cells, one_cell, step
from physics.engine.axiom_3_structure.logic.take_own import take_own


def test_the_total_is_exact_over_1000_steps_and_no_box_goes_below_0():
    store = one_cell()
    start = store.box_quanta.numpy().copy()
    for _ in range(1000):
        step(store)
        boxes = store.box_quanta.numpy()
        assert int(boxes.sum()) == 648_000 and boxes.min() >= 0
    assert store.step == 1000 and not np.array_equal(boxes, start)
    assert np.abs(store.box_phase.numpy()).max() <= np.float32(np.pi)


def test_a_cell_s_run_does_not_depend_on_the_slot_it_sits_in():
    alone, pair = calm_cells(2, own_cap=2), calm_cells(2, own_cap=2)
    take_own(alone, 0)                             # cell 0 in slot 0
    take_own(pair, 1)
    take_own(pair, 0)                              # cell 0 in slot 1
    for _ in range(50):
        step(alone)
        step(pair)
    assert np.array_equal(alone.box_quanta.numpy()[0], pair.box_quanta.numpy()[1])
    assert not np.array_equal(pair.box_quanta.numpy()[0], pair.box_quanta.numpy()[1])   # each cell has its own draws
    assert alone.box_quanta.numpy()[1].max() == 0  # a slot not in use is never visited
