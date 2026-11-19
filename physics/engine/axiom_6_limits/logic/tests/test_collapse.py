"""A joint past its limit flips to plus and merges with its two volumes; one below it stands."""
import numpy as np

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of

CALM, CELLS = 648_000, list(line_cells())


def _books(store, start: int) -> np.ndarray:
    """Every place's quanta, the books checked: boxes are totals, plus and minus are the start and what was made."""
    held, own = store.cell_quanta.numpy().astype(np.int64), store.own_cell.numpy()[:store.count]
    assert (store.box_quanta.numpy()[:store.count].astype(np.int64).sum(axis=1) == held[own]).all()
    assert np.abs(held).sum() == start + store.cell_made.numpy().astype(np.int64).sum()
    return held


def _fallen(steps: int = 30):
    """Two volumes of 400 calm volumes' quanta and a joint of 1: past the limit once the pull has arrived."""
    store, seen = three_cells(parts=(800, -2, 800)), []
    for _ in range(steps):
        before = store.cell_quanta.numpy().astype(np.int64)
        step(store)
        seen.append((store.collapses, before, _books(store, 801 * CALM)))
    return store, seen


def test_a_joint_below_the_limit_stands_and_one_past_it_flips_to_plus_and_merges():
    calm = three_cells()
    for _ in range(100):
        step(calm)
    assert calm.collapses == 0 and (calm.cell_quanta.numpy()[CELLS] != 0).all()  # pull 1.2: far below
    store, seen = _fallen()
    at = [fell for fell, _before, _after in seen].index(1)
    _fell, before, after = seen[at]
    assert at == 2  # the pull reaches the joint in the second step; the third finds it past
    assert store.flipped == -before[CELLS[1]] > 0  # minus lost
    assert after[after > 0].sum() - before[before > 0].sum() == store.flipped  # is plus gained
    assert sorted(after[CELLS]) == [0, 0, before[CELLS[0]] + before[CELLS[2]] - before[CELLS[1]]]
    assert store.collapses == 1 and (after[CELLS] == seen[-1][2][CELLS]).all()  # and it stays one


def test_the_merged_volume_is_one_whole_whirl():
    store, _seen = _fallen(steps=3)
    stands = int(np.argmax(store.cell_quanta.numpy()[CELLS]))  # where the fuller of the two stood
    assert swirl_of(store, CELLS[stands])["turns"] == (-1, -1, -1)
    assert (np.delete(store.box_quanta.numpy(), stands, axis=0) == 0).all()  # the other two slots: empty
