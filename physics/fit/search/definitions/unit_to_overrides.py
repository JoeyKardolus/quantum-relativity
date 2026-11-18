"""Unit to overrides: a point of the unit cube as engine settings.

Contract: `unit_to_overrides(unit)` takes one coordinate between 0 and 1 per name of
`SEARCHED`, in that order, and returns `{name: value}`: 0 is the low end of the name's
range and 1 the high end, evenly in the logarithm for a name of `LOG_RANGES` and evenly
for one of `LINEAR_RANGES`. Raises ValueError when the point has not one coordinate per
searched name.
"""
from __future__ import annotations

import math

from physics.fit.search.definitions.ranges import LINEAR_RANGES, LOG_RANGES, SEARCHED


def _place(name: str, coordinate: float) -> float:
    """The value of one searched setting at a coordinate of its range."""
    if name in LOG_RANGES:
        low, high = LOG_RANGES[name]
        return float(math.exp(math.log(low) + coordinate * (math.log(high) - math.log(low))))
    low, high = LINEAR_RANGES[name]
    return float(low + coordinate * (high - low))


def unit_to_overrides(unit) -> dict:
    """The engine settings a unit-cube point stands for."""
    if len(unit) != len(SEARCHED):
        raise ValueError(f"a point has {len(SEARCHED)} coordinates, got {len(unit)}")
    return {name: _place(name, coordinate) for name, coordinate in zip(SEARCHED, unit)}
