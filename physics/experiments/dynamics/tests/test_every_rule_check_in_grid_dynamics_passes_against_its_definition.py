"""Every rule check in Grid dynamics passes against its definition.

WHAT IT PINS. physics/experiments/dynamics through `rule_checks`: every check
passes, and every number with a known value matches it.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Drift h = 0.01
for 10 steps lengthens by 1.01^10 = 1.1046221254112045 (typed here, worked out
by hand); energy flow keeps the total (ratio 1); rewiring keeps the count.

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. The drift check's known value computed as
(1 + h) * n instead of (1 + h)^n.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

from physics.experiments.dynamics.checks import rule_checks


def test_every_rule_check_in_grid_dynamics_passes_against_its_definition() -> None:
    """All checks pass; drift's known value is 1.01^10."""
    numbers, passed = rule_checks(3)
    assert len(passed) == 6 and all(passed)
    drift = next(entry for entry in numbers if entry["name"].startswith("drift"))
    assert abs(drift["known"] - 1.1046221254112045) < 1e-6
    for entry in numbers:
        if entry["known"] is not None:
            assert abs(entry["measured"] - entry["known"]) < 1e-6, entry["name"]
