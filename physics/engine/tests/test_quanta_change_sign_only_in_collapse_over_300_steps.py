"""Quanta never change sign except in joint collapse, over 300 steps of the default rules.

Bible rule 4: plus and minus never cancel, and joint collapse is the one place where
pushing energy becomes pulling energy. The default start shape (seed 1, 5**3 cells of
10**3 boxes, new_universe) runs 300 steps with the default settings. After every step:
the plus quanta have grown by exactly the minus quanta collapse turned plus (tallies[0,
3]) and the minus quanta fallen by the same, the amount ignoring sign is the start's,
every box that changed kind is one a rule marked (step.py: 1 collapse, 2 claim; minus to
plus by either, plus to minus and empty to minus by a claim only), no box empties, and
every 50 steps no plus box has an empty box or the wall among its 26 neighbours.
"""
import numpy as np

from physics.engine import Settings, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu


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
def test_only_collapse_turns_minus_quanta_plus_and_every_kind_change_has_a_rule():
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
            spread, collapsed, claims, _ = state.tallies.numpy()[0]
            print(f"step {count + 1}: plus {now['plus']} (start {start['plus']}), turned plus {turned}, "
                  f"collapsed boxes {collapsed}, claims {claims}, spread {spread}, kind changes {int(changed.sum())}")
            assert _unwrapped(state.quanta.numpy()) == 0
        before = after
