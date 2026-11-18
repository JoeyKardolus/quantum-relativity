"""Held shells: how far round a core energy flowed inward through every boundary.

Contract: `held_shells(inflow)` takes, per boundary 0 .. n, the net plus energy that
crossed it inward over the reading window (twin subtracted; boundary j closes the core
and shells 1 .. j) and returns how many shells the held region reaches: the largest h
such that every boundary 1 .. h took energy in (inflow > 0), never more than n - 1 (the
last boundary read is the limit of the reading, not a region). Boundary 0, the core's
own skin, is not asked: nothing is assumed about the core. 0 when boundary 1 lost energy.
"""
from __future__ import annotations


def held_shells(inflow) -> int:
    """The number of shells out to which every boundary from 1 on took energy in."""
    held = 0
    while held + 1 < len(inflow) - 1 and inflow[held + 1] > 0.0:
        held += 1
    return held
