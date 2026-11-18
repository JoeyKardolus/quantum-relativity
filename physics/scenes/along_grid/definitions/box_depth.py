"""Box depth: how much deeper a 1 / distance pull lies at one distance than at another, in a wrapping box.

Contract: `box_depth(radius, far, box, copies=4)` returns as a float the depth of the
potential 1 / distance at `radius` along a row less its depth at `far` along the same
row, from a point source and its copies `box` apart in every direction (`copies` each
way: the universe wraps, so a lump's copies pull too), with the even level taken out as
a wrapping box needs: a block of copies summed bare also pulls toward its own middle,
as an evenly filled cube would, and that part, (2 pi / 3) distance ** 2 / box ** 3, is
removed. Near the source it is 1 / radius - 1 / far; toward half the box the copies
flatten it, and it agrees with box_field along an axis. Distances are in any one unit
(the scenes use summed lengths). Raises ValueError unless 0 < radius, 0 < far and
box > 0.
"""
from __future__ import annotations

import math

import numpy as np


def box_depth(radius: float, far: float, box: float, copies: int = 4) -> float:
    """Depth of a 1 / distance pull at `radius` below that at `far`, from a point and its copies."""
    if not (radius > 0.0 and far > 0.0 and box > 0.0):
        raise ValueError(f"radius, far and box must be above 0, got {radius}, {far}, {box}")
    shifts = np.arange(-copies, copies + 1) * box
    east, north, up = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    depth = [(1.0 / np.sqrt((at + east) ** 2 + north ** 2 + up ** 2)).sum() for at in (radius, far)]
    even_level = (2.0 * math.pi / 3.0) * (radius ** 2 - far ** 2) / box ** 3
    return float(depth[0] - depth[1] + even_level)
