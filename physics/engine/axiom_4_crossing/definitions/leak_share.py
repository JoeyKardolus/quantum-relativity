"""Leak share: how much of a whirl leaks in a turn, and toward which of the directions round it.

A volume is a whirl, and the whirl leaks: the bigger the volume (the lower its energy), the
more. The one law: in a turn of its swirl a volume `size` calm volumes big leaks the share
`LEAK * size / (1 + LEAK * size)` of its quanta: in step with its size while it is small,
and never more than it holds. It leaks alike toward the 26 directions round it (6 sides,
12 edges, 8 corners): each box's leak of a step goes one of them, by the draw.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import beside

# The leak's one setting, chosen before any run: 1, so the calm volume is the mark and no
# new size comes in: a calm volume with nothing round it leaks half of itself in a turn.
LEAK = 1.0


@wp.func
def leak_share(quanta: wp.float64, calm: wp.float64) -> wp.float64:
    """The share of itself a volume that holds `quanta` leaks in a turn of its swirl."""
    size = wp.float64(LEAK) * calm / quanta
    return size / (wp.float64(1.0) + size)


@wp.func
def toward(cell: wp.int32, side: wp.int32, uniform: wp.float64) -> wp.int32:
    """The volume's place next to `cell` in the direction the draw picks; -1 past the room's end."""
    pick = wp.int32(uniform * wp.float64(26.0))
    if pick >= 13:
        pick += 1  # 13 of the 27 is the volume itself
    place = beside(cell, side, 0, 2 * (pick / 9 - 1))
    if place >= 0:
        place = beside(place, side, 1, 2 * (pick / 3 % 3 - 1))
    if place >= 0:
        place = beside(place, side, 2, 2 * (pick % 3 - 1))
    return place
