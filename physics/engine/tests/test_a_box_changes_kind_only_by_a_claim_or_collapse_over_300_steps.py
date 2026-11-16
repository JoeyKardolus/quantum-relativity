"""A box changes kind only when a cell claims it (its energy first flowing out to its old owner) or by collapse.

Owner, 2026-11-16 (decision C): Planck boxes are only the computing grid; a box changes
owner, and so kind, when a cell claims it, and before it changes kind its energy flows
out to its old owner, so no box ever holds two signs; volumes grow and shrink by their
energy. Bible rule 4: joint collapse is the one place where minus quanta turn plus.
The default start shape (seed 1, 5**3 cells of 10**3 boxes, new_universe) runs 300
steps with the default settings. After every step: the amount ignoring sign is the
start's; the plus quanta have grown by exactly the minus quanta collapse turned plus
(tallies[0, 3]) and the minus quanta fallen by the same, so a claim never turns a
quantum's sign (the claimed box's quanta went to boxes of its old kind); every box that
changed kind is one a rule marked (step.py: 1 collapse, 2 claim; minus to plus by
either, plus to minus and empty to minus by a claim only); no box empties. Every 50
steps no plus box has an empty box or the wall among its 26 neighbours (every energy
volume stays wrapped) and the joint boxes form one face-connected web (decision A).
At the end cells() finds no two joints between the same two energy volumes; how many
joints hold boxes apart from their main slab and how many leak crossings run through
their own joint (crossings.py) are printed, not judged.
"""
import numpy as np
from scipy import ndimage

from physics.engine import JOINT, Settings, cells, host_boxes, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu
from physics.engine.tests.crossings import crossings


def _unwrapped(quanta: np.ndarray) -> int:
    """How many plus boxes have an empty box (or the wall) among their 26 neighbours."""
    padded = np.pad(quanta, 1)
    side = quanta.shape[0]
    empty_near = np.zeros(quanta.shape, dtype=bool)
    for shift in np.ndindex(3, 3, 3):
        if shift != (1, 1, 1):
            empty_near |= padded[shift[0]:shift[0] + side, shift[1]:shift[1] + side, shift[2]:shift[2] + side] == 0
    return int(((quanta > 0) & empty_near).sum())


@needs_gpu
def test_only_claims_and_collapse_change_a_box_kind_and_no_quantum_changes_sign_but_by_collapse():
    state = new_universe(cells_per_side=5, boxes_per_cell_side=10, seed=1)
    settings = Settings()
    start = totals(state)
    before = np.sign(state.quanta.numpy())
    for count in range(300):
        step(state, settings)
        now = totals(state)
        turned = int(state.tallies.numpy()[0, 3])
        after = np.sign(state.quanta.numpy())
        marks = state.marks.numpy()
        changed = before != after
        to_plus = changed & (after > 0)
        to_minus = changed & (after < 0)
        assert now["amount"] == start["amount"]
        assert now["plus"] - start["plus"] == turned
        assert start["minus"] - now["minus"] == turned
        assert not np.any(changed & (after == 0))
        assert np.all(marks[to_plus] != 0) and np.all(before[to_plus] < 0)
        assert np.all(marks[to_minus] & 2)
        if count % 50 == 49:
            quanta = state.quanta.numpy()
            spread, collapsed, claims, _ = state.tallies.numpy()[0]
            web = ndimage.label(quanta < 0)[1]
            print(f"step {count + 1}: plus {now['plus']} (start {start['plus']}), turned plus {turned}, "
                  f"collapsed boxes {collapsed}, claims {claims}, spread {spread}, kind changes {int(changed.sum())}, "
                  f"web pieces {web}")
            assert _unwrapped(quanta) == 0
            assert web == 1
        before = after
    quanta = state.quanta.numpy()
    found = cells(host_boxes(state))
    joints = np.flatnonzero(found["kind"] == JOINT)
    pairs = [tuple(pair) for pair in found["between"][joints].tolist()]
    split = sum(1 for joint in joints if ndimage.label(found["owner"] == joint)[1] > 1)
    through, total = crossings(quanta, found)
    print(f"volumes {found['volumes']}, joints {joints.size}, joints in more than one piece {split}, "
          f"crossings between two volumes {total}, through their own joint {through}")
    assert len(set(pairs)) == len(pairs)
