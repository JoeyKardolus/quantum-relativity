"""Allowed range of every control the page can set.

Contract: `setting_ranges(calibrated)` returns {name: (minimum, maximum)} for
every CellSettings field plus "unevenness" and "speed". A rule's range runs
from 0 to SCALED times its calibrated value (a negative value flips the range;
a zero value gets 0 to ZERO_REACH), since its sensible size depends on the
rules' own units. Leak and joint push are shares per step, so they never
go past 1, and push_limit stays below 1 (at 1 a joint's squeeze divides by zero).
Unevenness and speed are fixed.
"""
from __future__ import annotations

from dataclasses import fields

from physics.cells import CellSettings

SCALED = 5.0
ZERO_REACH = 0.01
_CAPS = {"leak": 1.0, "joint_push": 1.0, "push_limit": 0.99}
_FIXED = {
    "unevenness": (0.0, 1.0),
    "speed": (0.5, 60.0),
}


def setting_ranges(calibrated: CellSettings) -> dict[str, tuple[float, float]]:
    """Minimum and maximum per control, rules scaled to their calibrated values."""
    ranges = {}
    for field in fields(CellSettings):
        reach = SCALED * float(getattr(calibrated, field.name)) or ZERO_REACH
        if field.name in _CAPS:
            reach = min(reach, _CAPS[field.name])
        ranges[field.name] = (min(0.0, reach), max(0.0, reach))
    ranges.update(_FIXED)
    return ranges
