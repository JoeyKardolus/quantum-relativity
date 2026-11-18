"""Newton tilt: the tilt a pull of one over distance squared gives, measured as the scene measures the engine's pull.

Contract: `newton_tilt(distance, reach, box, copies=4)` returns as a float the
difference of the potential 1 / distance between the places `reach` short of and
`reach` beyond a point at `distance`, over 2 * reach: from a point source and its
copies `box` apart in every direction (`copies` each way; the universe wraps), with the
even level taken out as a wrapping box needs (a block of copies summed bare also pulls
toward its own middle; that part, (4 pi / 3) distance / box ** 3, is removed). Far from
the copies it is 1 / distance ** 2; half the box from the source it is 0, the source
and its copy pulling alike. All four in one unit (summed lengths). Raises
ValueError unless 0 < reach < distance and box > 0.
"""
from __future__ import annotations

import math

import numpy as np


def newton_tilt(distance: float, reach: float, box: float, copies: int = 4) -> float:
    """Tilt of a 1 / distance potential over `reach` either side of a place `distance` from the source."""
    if not (0.0 < reach < distance and box > 0.0):
        raise ValueError(f"need 0 < reach < distance and box > 0, got {reach}, {distance}, {box}")
    shifts = np.arange(-copies, copies + 1) * box
    east, north, up = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    depth = [(1.0 / np.sqrt((distance + offset + east) ** 2 + north ** 2 + up ** 2)).sum() for offset in (-reach, reach)]
    return float((depth[0] - depth[1]) / (2.0 * reach) - (4.0 * math.pi / 3.0) * distance / box ** 3)
