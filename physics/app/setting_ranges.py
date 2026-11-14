"""Allowed range of every control the page can set.

Contract: `setting_ranges(calibrated)` returns {name: (minimum, maximum)} for
every RuleSettings field plus "unevenness" and "speed". Attraction and drift
are scaled to the reset's calibrated values (0 to SCALED times them; a
negative drift flips the range), since their sensible size depends on the
network; every other range is fixed.
"""
from __future__ import annotations

from physics.rules import RuleSettings

SCALED = 5.0
_FIXED = {
    "energy_length": (0.0, 1.0),
    "energy_flow": (0.0, 1.0),
    "joint_push": (0.0, 2.0),
    "push_limit": (0.0, 0.1),
    "repulsion": (0.0, 1.0),
    "rewiring": (0, 500),
    "target_degree": (1.0, 20.0),
    "target_triangles": (0.0, 20.0),
    "triangle_weight": (0.0, 5.0),
    "unevenness": (0.0, 1.0),
    "speed": (0.5, 60.0),
}


def setting_ranges(calibrated: RuleSettings) -> dict[str, tuple[float, float]]:
    """Minimum and maximum per control, attraction and drift scaled to the calibration."""
    ranges = dict(_FIXED)
    for name in ("attraction", "drift"):
        reach = SCALED * getattr(calibrated, name) or 0.01
        ranges[name] = (min(0.0, reach), max(0.0, reach))
    return ranges
