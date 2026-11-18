"""Ring box: the boxes round a swirl's axis, in the order the swirl goes round them.

Contract: a cell's block, cut across its swirl axis, is 6 layers of 6 by 6 boxes. Each
layer is three square rings round the axis (which runs between the boxes, at 2.5, 2.5):
ring 0 the 20 boxes on the outside, ring 1 the 12 inside those, ring 2 the 4 in the
middle. Every box of the block is on exactly one ring of one layer.
  RINGS                          3
  ring_size_host(ring) -> int    20, 12, 4
  ring_box_host(axis, layer, ring, index) -> (x, y, z)
      box number `index` (taken modulo the ring's size) of that ring, going round in the
      sense that turns the first coordinate across the axis toward the second (axis 0: y
      toward z, axis 1: x toward z, axis 2: x toward y): the way a swirl of sense +1
      goes round
Host only: the engine's kernels never walk a ring (the fluid goes round by its own
phase, swirl_phase.py); a ring is how the going round is read. This is book-keeping
inside one cell: it gives no box a place in space.
"""
from __future__ import annotations

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK

RINGS = BLOCK // 2


def ring_size_host(ring: int) -> int:
    """How many boxes a ring has."""
    return 4 * (BLOCK - 1 - 2 * int(ring))


def ring_box_host(axis: int, layer: int, ring: int, index: int) -> tuple:
    """Box number `index` of one ring of one layer, as (x, y, z)."""
    side = BLOCK - 1 - 2 * int(ring)
    place = int(index) % (4 * side)
    first = second = int(ring)
    if place < side:
        first = ring + place
    elif place < 2 * side:
        first, second = ring + side, ring + place - side
    elif place < 3 * side:
        first, second = ring + 3 * side - place, ring + side
    else:
        second = ring + 4 * side - place
    return ((layer, first, second), (first, layer, second), (first, second, layer))[axis]
