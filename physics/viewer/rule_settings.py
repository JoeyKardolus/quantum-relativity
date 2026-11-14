"""RuleSettings for a scene, built against the shared v2 contract.

Contract: `rule_settings(strengths)` returns (RuleSettings, missing): the
settings with every strength whose field RuleSettings has, and the names of
the strengths it does not have yet (joint_push and push_limit before the v2
rules land). A scene reports `missing` on the page rather than pretending a
rule ran.
"""
from __future__ import annotations

from dataclasses import fields

from physics.rules import RuleSettings


def rule_settings(strengths: dict) -> tuple[RuleSettings, tuple[str, ...]]:
    """RuleSettings from the strengths it knows; the names it does not know."""
    known = {field.name for field in fields(RuleSettings)}
    missing = tuple(name for name in strengths if name not in known)
    return RuleSettings(**{name: value for name, value in strengths.items() if name in known}), missing
