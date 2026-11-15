"""Allowed range of every control the page can set.

Contract: `setting_ranges(calibrated)` returns {name: (minimum, maximum)} for
every EngineSettings field plus "unevenness" and "speed". A rule's range runs
from 0 to SCALED times its calibrated value (a negative value flips the range;
a zero value gets 0 to ZERO_REACH), since its sensible size depends on the
rules' own units. Leak and joint push are shares per step, so they never
go past 1, push_limit stays below 1 (at 1 a joint's squeeze divides by zero), and
seal_point stays below 1 (a squeeze depth; the engine caps it at 1 - push_limit),
and push_power stays at or below 4 (past it the joint step goes unstable).
Unevenness and speed are fixed.
"""
from __future__ import annotations

from dataclasses import fields

from physics.cells import EngineSettings

SCALED = 5.0
ZERO_REACH = 0.01
_CAPS = {"leak": 1.0, "joint_push": 1.0, "push_limit": 0.99, "seal_point": 0.99,
         "push_power": 4.0}
_FIXED = {
    "unevenness": (0.0, 1.0),
    "speed": (0.5, 60.0),
}


def setting_ranges(calibrated: EngineSettings) -> dict[str, tuple[float, float]]:
    """Minimum and maximum per control, rules scaled to their calibrated values."""
    ranges = {}
    for field in fields(EngineSettings):
        reach = SCALED * float(getattr(calibrated, field.name)) or ZERO_REACH
        if field.name in _CAPS:
            reach = min(reach, _CAPS[field.name])
        ranges[field.name] = (min(0.0, reach), max(0.0, reach))
    ranges.update(_FIXED)
    return ranges
