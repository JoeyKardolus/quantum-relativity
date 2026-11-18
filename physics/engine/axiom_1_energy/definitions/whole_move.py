"""Whole move: a flow made into whole quanta, so one whole number leaves and the same arrives.

`floor(flow + uniform)` is right on average; cut so no box goes below 0. It cannot be undone.
"""
import warp as wp


@wp.func
def whole_move(flow: wp.float64, uniform: wp.float64, giver_has: wp.int32, taker_has: wp.int32):
    """The whole quanta that go from giver to taker (below 0: the other way)."""
    return wp.clamp(wp.int32(wp.floor(flow + uniform)), -taker_has, giver_has)
