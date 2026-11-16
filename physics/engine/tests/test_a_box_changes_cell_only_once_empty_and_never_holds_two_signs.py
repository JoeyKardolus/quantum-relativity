"""A box changes cell only once its energy has flowed out, so no box ever holds two signs; only collapse turns minus plus.

Owner, 2026-11-16: the Planck boxes stay; when a cell's boxes follow its size a box
changes owner only after its energy has flowed out (exchange.py and mortar_shift.py
empty it into its own cell and refill it from the new one inside one move, so no box
ever holds two signs); plus and minus never cancel; only joint collapse turns minus
into plus and only a tear plus into minus. The start new_universe(5, 8, seed=1,
margin=8) runs 150 steps with the default settings. After every step: every box's sign
is its cell's kind (state.kind: plus in a volume, minus in a joint, 0 in empty space);
the amount ignoring sign is the start's; the plus quanta have grown by exactly the
minus quanta collapse turned plus less the plus quanta tears turned minus; and every box
that changed sign is collapse marked (bit 1), tear marked (bit 8) or changed cell (bit
2). Every 50 steps the joint boxes form one face-connected web and no plus box has an
empty box or a wall among its 26 neighbours (every volume stays wrapped). Boxes did
change cell.
"""
import numpy as np

from physics.engine import Settings, group_label, near, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu


@needs_gpu
def test_owner_changes_keep_every_box_on_its_cell_sign():
    state = new_universe(5, 8, seed=1, margin=8)
    settings = Settings()
    start = totals(state)
    before = np.sign(state.quanta.numpy())
    for count in range(150):
        step(state, settings)
        quanta = state.quanta.numpy()
        owner = state.owner.numpy()
        kinds = np.concatenate([state.kind[0].astype(np.int64), [0]])
        now = totals(state)
        turned = int(state.tallies.numpy()[0, 3]) - int(state.torn[0])
        after = np.sign(quanta)
        marks = state.marks.numpy()
        assert np.array_equal(after, kinds[owner])
        assert now["amount"] == start["amount"]
        assert now["plus"] - start["plus"] == turned and start["minus"] - now["minus"] == turned
        assert np.all(marks[before != after] & 11)
        before = after
        if count % 50 == 49:
            assert group_label(quanta < 0, state.wrap)[1] == 1
            assert not np.any((quanta > 0) & near(quanta == 0, state.wrap, wall=True))
            print(f"step {count + 1}: owner changes {int(state.tallies.numpy()[0, 2])}, turned plus {turned}")
    assert int(state.tallies.numpy()[0, 2]) > 0
