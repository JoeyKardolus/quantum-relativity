"""The whirl is in the fluid: one whole turn about each direction, at the start and after 2,000 steps."""
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


def test_the_whirl_counts_one_whole_turn_about_each_direction_at_the_start_and_after_2000_steps():
    store = one_cell()
    for steps in (0, 2000):
        while store.step < steps:
            step(store)
        seen = swirl_of(store, 0)
        assert seen["turns"] == (-1, -1, -1) and min(seen["round_flow"]) > 0, (steps, seen)
        assert seen["core"] < 0.5, (steps, seen)
        flows = np.array(seen["round_flow"])
        assert np.ptp(flows) < 0.15 * flows.mean(), (steps, seen)  # alike (measured: 0.07 at 2,000)


def test_fluid_with_no_swirl_reads_no_turn():
    flat, still = np.full(216, 3000), np.zeros(216)
    assert whole_turns(still, 2) == 0 and round_flow(flat, still, 2) == 0.0
    assert core_share(flat, 2) == 1.0 and off_rule(flat, flat) == 0.0
