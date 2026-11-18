"""Whole move: a flow made into a whole number of quanta that empties neither side past 0.

Contract: `whole_move(flux, uniform, giver_has, taker_side_has)` (Warp func) returns an
int64: `floor(flux + uniform)`, signed, `uniform` in [0, 1) from draw_uniform.py, so the
rounding is right on average and carries nothing over. Above 0 the quanta go from the
giver to the taker side, below 0 the other way. It is cut so that neither side goes past
0: never more than `giver_has`, never less than `-taker_side_has` (both counted ignoring
sign). The caller counts a cut: the move was cut when the result differs from
`floor(flux + uniform)`.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def whole_move(flux: wp.float64, uniform: wp.float64, giver_has: wp.int64, taker_side_has: wp.int64) -> wp.int64:
    """The whole quanta that move, giver to taker side above 0, never past what a side holds."""
    moved = wp.int64(wp.floor(flux + uniform))
    if moved > giver_has:
        moved = giver_has
    if moved < -taker_side_has:
        moved = -taker_side_has
    return moved
