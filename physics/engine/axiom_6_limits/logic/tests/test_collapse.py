"""A joint whose two volumes are both past the limit flips to plus where it stands; nothing else changes."""
import numpy as np

from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_3_structure.definitions.store import put
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_3_structure.logic.joints_of import joints_of
from physics.engine.axiom_5_packing.logic.staging import copy_of
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP
from physics.engine.axiom_6_limits.logic.collapse import collapse

CALM, CELLS = 648_000, list(line_cells())


def _books(store, start: int) -> np.ndarray:  # every place's quanta, boxes against totals, all against the start
    held, own = store.cell_quanta.numpy().astype(np.int64), store.own_cell.numpy()[:store.count]
    assert (store.box_quanta.numpy()[:store.count].astype(np.int64).sum(axis=1) == held[own]).all()
    assert np.abs(held).sum() == start + store.cell_made.numpy().astype(np.int64).sum()
    return held


def _pulled(parts: tuple, pulls: tuple):  # three cells, the pull at the two volumes set by hand, after one look
    store = three_cells(parts=parts)
    put(store.cell_pull, CELLS[0], pulls[0])
    put(store.cell_pull, CELLS[2], pulls[1])
    collapse(store)
    return store


def test_the_limit_is_the_same_for_a_thin_joint_and_a_fat_one_under_the_same_pull():
    for joint in (-1, -2, -400):  # 2 calm lengths thick, 1, and a two-hundredth
        assert _pulled((2, joint, 2), (DEEP, DEEP)).collapses == 1
        assert _pulled((2, joint, 2), (0.999 * DEEP, 0.999 * DEEP)).collapses == 0
        assert _pulled((2, joint, 2), (9 * DEEP, 0.999 * DEEP)).collapses == 0  # one side can still climb out


def test_no_number_grows_at_a_merge_and_a_cell_s_number_holds_more_than_32_bits():
    parts = (6000, -8000, 6000)  # the joint alone is past 2 to the power 31, the three are 6.5 billion
    store = _pulled(parts, (DEEP, DEEP))
    held, each = _books(store, 20_000 * 324_000), [part * 324_000 for part in (6000, 8000, 6000)]
    assert store.collapses == 1 and held[CELLS].tolist() == each  # the joint: plus now
    assert store.flipped == each[1] > 2 ** 31 and swirl_of(store, CELLS[1])["turns"] == (-1, -1, -1)
    for _ in range(25):
        step(store)
    assert _books(store, 20_000 * 324_000)[CELLS[1]] == each[1]  # it stays plus; its two leak nowhere


def _staged(store):
    """A grown patch whose first volume and one beside it are set past the limit: the store after
    the collapse, the quanta and boxes before it, the joint's slot and the two volumes' slots."""
    joints, (first, second), _axis = joints_of(store)
    one = int(np.flatnonzero(first == 0)[0])  # a joint the first volume (slot 0) lies before
    for slot in (0, int(second[one])):
        put(store.cell_pull, int(store.own_cell.numpy()[slot]), 2 * DEEP)
    before = store.cell_quanta.numpy().copy(), store.box_quanta.numpy().copy()
    collapse(store)
    return store, before, int(joints[one]), (0, int(second[one]))


def test_a_collapse_changes_only_its_joint_and_every_cell_keeps_its_place_and_its_number(patch):
    store, (quanta, boxes), joint, ends = _staged(patch)
    cell, held = int(store.own_cell.numpy()[joint]), _books(store, 2 ** 30)
    assert store.collapses == 1 and held[cell] == -quanta[cell] == store.flipped > 0  # minus lost is plus gained
    assert (np.delete(held, cell) == np.delete(quanta, cell)).all()
    assert (np.delete(store.box_quanta.numpy(), joint, axis=0) == np.delete(boxes, joint, axis=0)).all()
    assert swirl_of(store, cell)["turns"] == (-1, -1, -1)  # laid as a volume's whole whirl
    joints, (first, second), _axis = joints_of(store)
    assert joint not in joints and len(joints) == (quanta < 0).sum() - 1  # every other joint stands
    assert not ((np.isin(first, ends)) & (np.isin(second, ends))).any()  # none is left between the two


def test_the_books_stay_exact_and_the_same_start_gives_the_same_run_through_a_collapse(grown_patch):
    runs = [_staged(copy_of(grown_patch))[0] for _ in range(2)]
    for store in runs:
        for _ in range(12):
            step(store)
            _books(store, 2 ** 30)
        assert np.isfinite(store.cell_pull.numpy()).all()
    assert np.array_equal(runs[0].box_quanta.numpy(), runs[1].box_quanta.numpy())
    assert np.array_equal(runs[0].box_phase.numpy(), runs[1].box_phase.numpy())
