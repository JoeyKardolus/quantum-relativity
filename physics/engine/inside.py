"""Inside (Warp func): whether a box index lies in the same universe's cube as the box that reads it.

Contract: `inside(box, home, side)` is True when box is a box of universe `home`: the
universes of a batch sit side by side along the first axis (universe u holds first
indices u * side .. u * side + side - 1, state.py), so it checks home * side <= box[0] <
(home + 1) * side and 0 <= box[1], box[2] < side. Outside a universe's cube there is no
space: its walls are closed (no hop, no pull, no owner change across them), and no rule ever
reads a neighbouring universe. A box's own universe is its first index // side.
"""
from __future__ import annotations

import warp as wp


@wp.func
def inside(box: wp.vec3i, home: int, side: int) -> bool:
    """True when `box` is a box of universe `home`'s side**3 cube."""
    low = home * side
    return (box[0] >= low and box[0] < low + side and box[1] >= 0 and box[2] >= 0 and box[1] < side
            and box[2] < side)
