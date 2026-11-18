"""The swirl is in the fluid: one whole turn at the start and after 1,000 whole-quanta steps, core open."""
import numpy as np

from physics.engine import one_cell, step
from physics.engine.axiom_2_swirls.logic.read_swirl import (
    core_share, off_rule, ring_places, round_flow, whole_turns)
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of


def test_a_ring_goes_once_round_the_axis_through_neighbours():
    for ring, boxes in ((0, 20), (1, 12), (2, 4)):
        first, second = (np.array(places) for places in ring_places(6, ring))
        assert len(set(zip(first, second))) == boxes
        assert (np.abs(first - np.roll(first, -1)) + np.abs(second - np.roll(second, -1)) == 1).all()


def test_the_swirl_counts_one_whole_turn_at_the_start_and_after_1000_steps():
    store = one_cell()
    for steps in (0, 1000):
        while store.step < steps:
            step(store)
        seen = swirl_of(store, 0)
        assert seen["turns"] == -1 and seen["round_flow"] > 0, (steps, seen)
        assert seen["core"] < 0.5, (steps, seen)


def test_fluid_with_no_swirl_reads_no_turn():
    flat, still = np.full(216, 3000), np.zeros(216)
    assert whole_turns(still, 4) == 0 and round_flow(flat, still, 4) == 0.0
    assert core_share(flat, 4) == 1.0 and off_rule(flat, flat) == 0.0
