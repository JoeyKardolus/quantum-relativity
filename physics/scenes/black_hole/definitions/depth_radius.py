"""Depth radius: the distance from a core at which its pull's depth falls to a fixed level.

Contract: `depth_radius(depth, distance, level)` takes per shell 0 .. n the pull's depth
there (how much lower the pull lies than far away; twin subtracted) and the shell's
distance from the core's middle, and a level. Returns `(radius, deep)`: `deep` is the
first shell whose depth is below the level (n when none is) and `radius` the distance
at which the depth crosses the level, read on a straight line between shell deep - 1
and shell deep. deep 0 (the depth is under the level at the core itself) gives radius
0.0. When the two shells' depths are equal the radius is shell deep's distance. Raises
ValueError for lists of unequal length or none.
"""
from __future__ import annotations


def depth_radius(depth, distance, level: float) -> tuple[float, int]:
    """Where the depth falls to the level, and the first shell below it."""
    if not len(depth) or len(depth) != len(distance):
        raise ValueError("need one depth and one distance per shell")
    last = len(depth) - 1
    deep = next((shell for shell in range(len(depth)) if depth[shell] < level), last)
    if deep == 0:
        return 0.0, 0
    fall = depth[deep - 1] - depth[deep]
    if fall == 0.0:
        return float(distance[deep]), deep
    share = (depth[deep - 1] - level) / fall
    return float(distance[deep - 1] + share * (distance[deep] - distance[deep - 1])), deep
