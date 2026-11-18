"""Carried: where a tracer lies along two boxes after fluid has moved between them.

The tracer keeps as much of the two boxes' fluid behind it as it had. `line` runs from
0 to 2 along the pair: below 1 it is in the low box, from 1 on in the high one.
"""
import warp as wp


@wp.func
def carried(line: float, low_has: float, high_has: float, stays: float) -> float:
    """The tracer's new place along the pair when `stays` is the low box's fluid after."""
    before = wp.min(line, 1.0) * low_has + wp.max(line - 1.0, 0.0) * high_has  # fluid behind it
    if before < stays:
        line = before / stays
    else:
        line = 1.0 + (before - stays) / wp.max(low_has + high_has - stays, 1.0e-6)
    return wp.clamp(line, 0.0, 1.9999)  # never onto the far edge: that is the next cell's
