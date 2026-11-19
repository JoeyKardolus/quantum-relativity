"""Whole move: a flow made into whole quanta, so one whole number leaves and the same arrives.

A flow is seldom whole, and a quantum cannot be shared. Which way the odd quantum goes is
decided by the swirl's own oscillation at that moment. A whirl turns in whole steps, one
box round its core at a time: `odd_part` is how far the giving box's phase stands through
the step it is in, and `floor(flow + odd_part)` is right over a step. (Through the whole
turn instead, the rounding keeps time with the whirl and slowly heaps the fluid on one
side: measured, four tenths of a box's quanta in 10,000 steps.) Nothing is drawn by chance:
the same start gives the same run. Cut so no box goes below 0.
"""
import warp as wp


@wp.func
def odd_part(phase: wp.float32, steps: wp.int32) -> wp.float64:
    """How far a phase stands through one of the `steps` whole steps of its turn, from 0 up to 1."""
    part = wp.float64(phase) * wp.float64(steps) / wp.float64(wp.tau)
    return part - wp.floor(part)


@wp.func
def whole_move(flow: wp.float64, part: wp.float64, giver_has: wp.int32, taker_has: wp.int32):
    """The whole quanta that go from giver to taker (below 0: the other way)."""
    return wp.clamp(wp.int32(wp.floor(flow + part)), -taker_has, giver_has)
