"""A joint past its limit flips to plus and merges with its two volumes; one below it stands."""
import numpy as np

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_3_structure.logic.give_back import whole_turns_laid

CALM, CELLS = 648_000, list(line_cells())


def _books(store, start: int) -> np.ndarray:
    """Every place's quanta, the books checked: boxes are totals, plus and minus are the start and what was made."""
    held, own = store.cell_quanta.numpy().astype(np.int64), store.own_cell.numpy()[:store.count]
    assert (store.box_quanta.numpy()[:store.count].astype(np.int64).sum(axis=1) == held[own]).all()
    assert np.abs(held).sum() == start + store.cell_made.numpy().astype(np.int64).sum()
    return held


def _fallen(seed: int = 1, steps: int = 30):
    """Two volumes of 400 calm volumes' quanta and a joint of 1: past the limit once the pull has arrived."""
    store, seen = three_cells(seed=seed, parts=(800, -2, 800)), []
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
    assert after[CELLS].tolist() == [before[CELLS[0]] + before[CELLS[2]] - before[CELLS[1]], 0, 0]
    assert store.collapses == 1 and after[CELLS[0]] == seen[-1][2][CELLS[0]]  # and it stays one


def test_the_merged_volume_is_one_whole_whirl():
    store, _seen = _fallen(steps=3)
    assert swirl_of(store, CELLS[0])["turns"] == whole_turns_laid(int(store.cell_laid.numpy()[CELLS[0]]))
    assert (store.box_quanta.numpy()[1:] == 0).all()  # the joint's and the other volume's boxes are empty
