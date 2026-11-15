"""A passage: how much of A's flow a box lets in, by the joints (B) it holds.

Contract: a Warp function, usable inside kernels only. `a_passage(a, b, a_rest,
b_rest, floor)` with a, b the A- and B-energy in a box: 1 when the box holds A at or
below `floor` (nothing there for A to pack against), else min(1, (b / b_rest) / (a /
a_rest)): the box's B concentration over its A concentration, each against its own
rest. Energy passes from energy volume to energy volume only through a joint (bible
foundation 4), and A never packs against A with no B between (energy volumes touch
only joints): A cannot flow into a box short of its joint share, in proportion to the
shortfall (transport.py, the receiving box on each face; the push of the A held back
goes to the receiving box, as against a wall). At the start every box holds its joint
share (new_grid.py), passage 1. (Gating the sending box too was tried first: A left
behind by its B could never move while the pull sped it up, 380 boxes per time unit
and 2000 substeps per step by step 90 at side 16, measured.)
"""
from __future__ import annotations

import warp as wp


@wp.func
def a_passage(a: float, b: float, a_rest: float, b_rest: float, floor: float) -> float:
    """1, or the box's B-to-A concentration ratio where it holds less than its joint share."""
    if a <= floor:
        return 1.0
    return wp.min(1.0, (b / b_rest) / (a / a_rest))
