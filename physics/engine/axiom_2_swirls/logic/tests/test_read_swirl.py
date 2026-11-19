"""The whirl is in the fluid: whole turns in each ring, all of it moving, the three alike, after 3,000 steps."""
import numpy as np

from physics.engine import one_cell, step
from physics.engine.axiom_2_swirls.definitions.ring_boxes import cap_boxes
from physics.engine.axiom_2_swirls.logic.read_swirl import (
    off_rule, round_flow, rows_of, still_share, whole_turns)
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of

ALIKE = 0.02  # the stated tolerance: the three rings' flows within 2% (measured: 0.4% at 3,000)


def test_a_ring_s_rows_go_once_round_its_core_and_close():
    boxes = np.arange(216)
    for ring in range(3):
        rows = rows_of(boxes, ring)
        assert rows.shape == (6, 12) and (rows // 72 == ring).all()  # a third of the block
        assert (np.roll(rows, -1, axis=0) - rows)[:-1].min() == 6  # the next box round the core
        assert ((rows[-1] - rows[0]) == 30).all()  # and the last closes on the first


def test_the_whirl_winds_whole_turns_in_each_ring_at_the_start_and_after_3000_steps():
    store = one_cell()
    for steps in (0, 3000):
        while store.step < steps:
            step(store)
        seen = swirl_of(store, 0)
        assert seen["turns"] == (-1, -1, -1), (steps, seen)
        assert seen["still"] == 0.0, (steps, seen)  # no link round a core stands or goes back
        flows = np.array(seen["round_flow"])
        assert flows.min() > 0 and np.ptp(flows) < ALIKE * flows.mean(), (steps, seen)
    assert seen["off_rule"] < 1.0  # measured: 0.21, under plain counting noise


def test_the_fluid_is_in_motion_over_the_whole_outside_after_3000_steps():
    store = one_cell()
    for _ in range(3000):
        step(store)
    quanta = store.box_quanta.numpy()[0]
    outside = quanta.reshape(3, 2, 6, 6)[:, -1]  # the last layer of each ring
    assert outside.min() > 0.9 * 3000 and quanta.min() > 0.9 * 3000  # no empty box, no empty line
    for axis in range(3):
        for after in (0, 1):
            assert quanta[cap_boxes(6, axis, after)].min() > 0.9 * 3000  # all six caps


def test_fluid_with_no_swirl_reads_no_turn_and_stands_still():
    flat, still = np.full(216, 3000), np.zeros(216)
    assert whole_turns(still, 2) == 0 and round_flow(flat, still, 2) == 0.0
    assert still_share(flat, still) == 1.0 and off_rule(flat, flat) == 0.0
    torn = still.copy()
    torn[72:78] = 2.0  # one row of ring 1 out of step with the others
    assert whole_turns(np.cumsum(np.ones(216)) * 0 + torn, 1) == 0 and whole_turns(torn, 0) == 0
