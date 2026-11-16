"""A box changes cell only once its energy has flowed out, so no box ever holds two signs; only collapse turns minus plus.

Owner, 2026-11-16: the Planck boxes stay, each belongs to the cell whose moving region
covers it, and when corners move a box changes owner only after its energy has flowed
out (no box ever holds two signs); plus and minus never cancel; only joint collapse
turns minus into plus. The default start shape (new_universe, seed 1, 5**3 cells of
8**3 boxes, margin 8) runs 150 steps with the default settings, with every hand over
(hand_over.py) watched: each box whose owner it changes held 0 quanta just before
(drain.py emptied it), and after it holds the sign of its new cell's kind. After every
step: every box's sign is its cell's kind (lattice_kinds: plus in a volume, minus in a
joint, 0 in the ring of empty space); the amount ignoring sign is the start's; the plus
quanta have grown by exactly the minus quanta collapse turned plus (tallies[0, 3]); and
every box that changed sign is one collapse marked (bit 1) or that changed cell (bit 2).
Every 50 steps the joint boxes form one face-connected web and no plus box has an empty
box or a wall among its 26 neighbours (every volume stays wrapped).
"""
import importlib

import numpy as np
from scipy import ndimage

from physics.engine import Settings, lattice_kinds, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu

step_module = importlib.import_module("physics.engine.step")


@needs_gpu
def test_owner_changes_only_on_empty_boxes_and_every_box_keeps_its_cell_sign(monkeypatch):
    state = new_universe(5, 8, seed=1, margin=8)
    kinds = lattice_kinds(5).ravel()
    settings = Settings()
    original = step_module.hand_over
    watched = {"changes": 0, "full": 0, "wrong_sign": 0}

    def watched_hand_over(watched_state, phase_index):
        before_quanta = watched_state.quanta.numpy()
        before_owner = watched_state.owner.numpy()
        original(watched_state, phase_index)
        after_owner = watched_state.owner.numpy()
        after_quanta = watched_state.quanta.numpy()
        moved = after_owner != before_owner
        watched["changes"] += int(moved.sum())
        watched["full"] += int((moved & (before_quanta != 0)).sum())
        watched["wrong_sign"] += int((moved & (np.sign(after_quanta) != kinds[after_owner])).sum())

    monkeypatch.setattr(step_module, "hand_over", watched_hand_over)
    start = totals(state)
    before = np.sign(state.quanta.numpy())
    for count in range(150):
        step(state, settings)
        quanta = state.quanta.numpy()
        owner = state.owner.numpy()
        now = totals(state)
        turned = int(state.tallies.numpy()[0, 3])
        after = np.sign(quanta)
        marks = state.marks.numpy()
        assert np.array_equal(after, kinds[owner])
        assert now["amount"] == start["amount"]
        assert now["plus"] - start["plus"] == turned and start["minus"] - now["minus"] == turned
        assert np.all(marks[before != after] & 3)
        before = after
        if count % 50 == 49:
            open_space = np.pad(quanta == 0, 1, constant_values=True)
            near_empty = ndimage.binary_dilation(open_space, structure=np.ones((3, 3, 3), dtype=bool))[1:-1, 1:-1, 1:-1]
            assert ndimage.label(quanta < 0)[1] == 1
            assert not np.any((quanta > 0) & near_empty)
            print(f"step {count + 1}: owner changes {watched['changes']}, of boxes still holding quanta "
                  f"{watched['full']}, with the wrong sign after {watched['wrong_sign']}, turned plus {turned}")
    assert watched["changes"] > 0
    assert watched["full"] == 0 and watched["wrong_sign"] == 0
