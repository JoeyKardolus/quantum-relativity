"""Allowed range of every control the page can set.

Contract: `setting_ranges(calibrated)` returns {name: (minimum, maximum)} for every
numeric field of the engine's `Settings` dataclass (`physics.abab.Settings`; booleans are
left out) plus "unevenness" and "speed". A rule's range runs from 0 to SCALED times its
calibrated value (a negative value flips the range; a zero value gets 0 to ZERO_REACH),
since its sensible size depends on the rules' own units. Where these names exist: leak and
joint_push are shares per step, so they never go past 1; push_limit and seal_point stay
below 1 (squeeze depths); push_power stays at or below 4 (a steeper wall needs more joint
substeps); split_size starts at 1 (the start's mean size: near 0 every volume splits every
step until memory runs out). Unevenness and speed are fixed.
"""
from __future__ import annotations

from dataclasses import fields

SCALED = 5.0
ZERO_REACH = 0.01
_CAPS = {"leak": 1.0, "joint_push": 1.0, "push_limit": 0.99, "seal_point": 0.99, "push_power": 4.0}
_FLOORS = {"split_size": 1.0}
_FIXED = {
    "unevenness": (0.0, 1.0),
    "speed": (0.5, 60.0),
}


def setting_ranges(calibrated) -> dict[str, tuple[float, float]]:
    """Minimum and maximum per control, rules scaled to their calibrated values."""
    ranges = {}
    for field in fields(calibrated):
        value = getattr(calibrated, field.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        reach = SCALED * float(value) or ZERO_REACH
        if field.name in _CAPS:
            reach = min(reach, _CAPS[field.name])
        lowest = max(min(0.0, reach), _FLOORS.get(field.name, -float("inf")))
        ranges[field.name] = (lowest, max(lowest, reach))
    ranges.update(_FIXED)
    return ranges
