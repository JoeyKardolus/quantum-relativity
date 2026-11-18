"""Limits: the three things that can stop a search going to a larger size.

`limit_law(probes, min_rate)` names the one that did:
  SPEED      some probe fitted and ran below the rate
  MEMORY     no probe was too slow, and some probe did not fit
  SIZE_LIST  every probe fitted and met the rate: the list ended first
Speed is named before memory: a size too slow is a result about the engine, a size that
did not fit may only say another program held the GPU.
"""
from __future__ import annotations

from typing import Sequence

from physics.bench.largest_universe.definitions.pass_mark import meets_rate

SPEED = "speed"
MEMORY = "memory"
SIZE_LIST = "the size list"


def limit_law(probes: Sequence, min_rate: float) -> str:
    """What stopped the search: speed, memory or the end of the size list."""
    if any(probe.fits and not meets_rate(probe.steps_per_second, min_rate) for probe in probes):
        return SPEED
    if any(not probe.fits for probe in probes):
        return MEMORY
    return SIZE_LIST
