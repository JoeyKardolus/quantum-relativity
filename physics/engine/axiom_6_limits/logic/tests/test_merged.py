"""A merged volume holds all three amounts, keeps its two volumes' other joints, and runs on with exact books."""
import numpy as np

from physics.engine import step
from physics.engine.axiom_3_structure.definitions.store import put
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_3_structure.logic.give_back import whole_turns_laid
from physics.engine.axiom_3_structure.logic.shared_sides import joints_of
from physics.engine.axiom_5_packing.logic.lumps import grown
from physics.engine.axiom_6_limits.logic.collapse import collapse
from physics.engine.axiom_6_limits.logic.tests.test_collapse import _books, _fallen


def _staged():
    """A cluster two layers out; the joint from the first volume along direction 0 is set past its limit."""
    store = grown(2)
    joints, (first, second), _axis = joints_of(store)
    one = int(np.flatnonzero(first == 0)[0])  # a joint the first volume (slot 0) lies before
    ends = (0, int(second[one]))
    mine = (np.isin(first, ends) | np.isin(second, ends)).sum()
    put(store.cell_pull, int(store.own_cell.numpy()[joints[one]]), 1e9)
    before = store.cell_quanta.numpy().astype(np.int64)
    collapse(store)
    return store, before, int(joints[one]), ends, int(mine)


def test_the_merged_volume_holds_all_three_and_keeps_its_two_volumes_other_joints():
    store, before, joint, ends, mine = _staged()
    own, held = store.own_cell.numpy(), _books(store, 2 ** 30)
    keep = int(store.own_whole.numpy()[ends[1]])
    assert keep in ends and store.own_whole.numpy()[ends[0]] == keep  # one volume, where the fuller stood
    assert held[own[keep]] == before[own[list(ends)]].sum() - before[own[joint]] and store.collapses == 1
    assert sorted(held[own[[joint, *ends]]])[:2] == [0, 0] and store.flipped == -before[own[joint]]
    joints, (first, second), _axis = joints_of(store)
    assert ((first == keep) | (second == keep)).sum() == mine - 1  # every other joint of the two
    shared = store.own_part.numpy()[joints]
    assert (shared[:, [1, 3]].max(axis=1) == 2).sum() == 8  # four sides hold two joints each now
    assert swirl_of(store, int(own[keep]))["turns"] == whole_turns_laid(int(store.cell_laid.numpy()[own[keep]]))


def test_the_books_stay_exact_and_the_same_seed_gives_the_same_run_through_collapses():
    runs = [_staged()[0] for _ in range(2)]
    for store in runs:
        for _ in range(12):
            step(store)
            _books(store, 2 ** 30)
        assert np.isfinite(store.cell_pull.numpy()).all()
    assert np.array_equal(runs[0].box_quanta.numpy(), runs[1].box_quanta.numpy())
    assert np.array_equal(runs[0].box_phase.numpy(), runs[1].box_phase.numpy())
    first, again, other = (_fallen(seed)[0].box_quanta.numpy() for seed in (1, 1, 2))
    assert np.array_equal(first, again) and not np.array_equal(first, other)
