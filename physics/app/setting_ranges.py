"""Allowed range of every control the page can set.

Contract: `setting_ranges(calibrated)` returns {name: (minimum, maximum)} for
every CellSettings field plus "unevenness" and "speed". A rule's range runs
from 0 to SCALED times its calibrated value (a negative value flips the range;
a zero value gets 0 to ZERO_REACH), since its sensible size depends on the
rules' own units. Energy flow is a share moved per step, so it never goes past
1. Unevenness and speed are fixed.
"""
from __future__ import annotations

from dataclasses import fields

from physics.cells import CellSettings

SCALED = 5.0
ZERO_REACH = 0.01
_SHARES = ("energy_flow",)
_FIXED = {
    "unevenness": (0.0, 1.0),
    "speed": (0.5, 60.0),
}


def setting_ranges(calibrated: CellSettings) -> dict[str, tuple[float, float]]:
    """Minimum and maximum per control, rules scaled to their calibrated values."""
    ranges = {}
    for field in fields(CellSettings):
        reach = SCALED * float(getattr(calibrated, field.name)) or ZERO_REACH
        if field.name in _SHARES:
            reach = min(reach, 1.0)
        ranges[field.name] = (min(0.0, reach), max(0.0, reach))
    ranges.update(_FIXED)
    return ranges
