"""Groups: far volumes read as one body, so the pull need not go pair by pair.

The volumes' places make a block, `wide` a direction. Level 0 is the volumes themselves; a
group of level 1 is the eight (two a direction) that lie together, a group of level 2
eight of those, and so on. A group is one body: its energy is its volumes' sum, it stands
at the middle of their energy, and it keeps its own past, so its pull arrives as late as
its distance says. A volume reads one by one the volumes of its own group and of the
`near` groups round it each way; further off it reads groups, each level as far out as the
`near` groups round its own group of the level above, and the top level to the end.
`near` 0 is every volume by itself: all pairs, the law with nothing left out. What reading
by groups leaves out is measured against that (`tests/test_groups.py`).
"""
import warp as wp


def wide_of(side: int) -> int:
    """How many volumes' places lie along a direction of a room `side` places wide."""
    return (side + 1) // 2


def top_of(side: int, near: int) -> int:
    """The highest level of groups: the first whose groups all lie near each other."""
    level = 0
    while near and (wide_of(side) + (1 << level) - 1) >> level > 2 * (2 * near + 1):
        level += 1
    return level


def nodes_of(side: int, top: int) -> int:
    """How many volumes' places and groups there are, all levels together."""
    return sum(((wide_of(side) + (1 << level) - 1) >> level) ** 3 for level in range(top + 1))


@wp.func
def across_of(wide: wp.int32, level: wp.int32) -> wp.int32:
    """How many groups of `level` lie along a direction."""
    return (wide + (1 << level) - 1) >> level


@wp.func
def first_of(wide: wp.int32, level: wp.int32) -> wp.int32:
    """The number of the first group of `level`: the levels are kept one after the other."""
    first = wp.int32(0)
    for lower in range(level):
        first += across_of(wide, lower) * across_of(wide, lower) * across_of(wide, lower)
    return first


@wp.func
def node_of(wide: wp.int32, level: wp.int32, at: wp.vec3i) -> wp.int32:
    """The number of the group of `level` that lies `at` along the three directions."""
    across = across_of(wide, level)
    return first_of(wide, level) + (at[0] * across + at[1]) * across + at[2]


@wp.func
def lead_node(wide: wp.int32, cell: wp.int32) -> wp.int32:
    """The level 0 number of the volume at place `cell`."""
    side = 2 * wide - 1
    return (cell / (side * side) / 2 * wide + cell / side % side / 2) * wide + cell % side / 2
