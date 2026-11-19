"""A universe moved into a bigger store is the same universe: it runs on as one grown there from the start."""
import numpy as np
import pytest

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.logic.roomier import roomier
from physics.engine.axiom_5_packing.logic.pull import pull

ARRAYS = ("cell_quanta", "cell_made", "cell_flow", "cell_pull", "box_quanta", "box_phase", "own_cell")


def test_a_universe_whose_room_filled_runs_on_in_a_bigger_one_box_for_box():
    small, roomy = first_volume(layers=1), first_volume(layers=3)
    with pytest.raises(RuntimeError, match="no room"):
        for _ in range(200):
            step(small)
    assert small.count == 1 and 0 < small.step < 200  # the first tear found no room, and changed nothing
    moved = roomier(small, 3)
    for _ in range(200 - moved.step):
        step(moved)
    for _ in range(200):
        step(roomy)
    assert moved.count == roomy.count == 81 and moved.step == roomy.step
    for name in ARRAYS:
        assert np.array_equal(getattr(moved, name).numpy(), getattr(roomy, name).numpy()), name


def test_moved_while_reading_by_groups_every_volume_lies_as_deep_as_before_and_the_books_are_the_same():
    store = first_volume(layers=3, near=1)
    for _ in range(600):
        step(store)
    moved = roomier(store, 6)
    before, plus = store.cell_pull.numpy().copy(), store.cell_quanta.numpy()
    assert moved.count == store.count > 81 and plus.sum() == moved.cell_quanta.numpy().sum()
    for one in (store, moved):
        pull(one)
    deep, there = store.cell_pull.numpy(), moved.cell_pull.numpy()
    assert sorted(there[there > 0]) == pytest.approx(sorted(deep[deep > 0]), rel=1e-3)  # other groups: a thousandth
    assert sorted(there[there > 0]) == pytest.approx(sorted(before[before > 0]), rel=0.01)  # the past came along: no jump
