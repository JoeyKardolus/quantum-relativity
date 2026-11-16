"""One group (Warp func): whether marked boxes round a Planck box form one connected group.

Contract: `one_group(members, adjacency)` takes a bit mask over a box's 3 x 3 x 3 block
(bit i for block place (i // 9, i // 3 % 3, i % 3), the centre 13 left out) and the
per-place neighbour masks of block_adjacency.py (26-neighbours, or faces only), and
returns True when the set bits are one connected group inside the block, or none.
drain.py reads it so a box leaving its cell never tears a cell or the joint web apart
(the digital-topology simple-point test).
"""
from __future__ import annotations

import warp as wp


@wp.func
def one_group(members: int, adjacency: wp.array(dtype=wp.int32)) -> bool:
    """True when the set bits of `members` are one connected group in the block, or none."""
    if members == 0:
        return True
    first = int(0)
    for index in range(27):
        if ((members >> index) & 1) == 1 and first == 0:
            first = 1 << index
    reached = first
    for _ in range(27):
        grown = reached
        for index in range(27):
            if ((reached >> index) & 1) == 1:
                grown = grown | (adjacency[index] & members)
        if grown == reached:
            return reached == members
        reached = grown
    return reached == members
