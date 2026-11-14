"""All rules on: the settings every scene starts from (owner, 2026-11-14).

Contract: `all_rules_on(**overrides)` returns RuleSettings with every rule
switched on except rewiring (0, only the space test rewires) and drift (0,
to be filled from `balancing_drift`). Keyword overrides replace any field.
Values:
  energy_length 1.0    the energy-length rule restored exactly every step
  energy_flow   0.1    energy diffuses between connections sharing a joint
  attraction    3e-5   log-shortening ~0.01 per step at the background pull
                       of a 6,000-joint 3D reference network (pull ~335);
                       pull grows with energy density, so a scene sets it as
                       its own target shortening / its background pull
  joint_push    0.5    a joint pushes back half the squeeze it gets
  push_limit    0.01   twice the push a joint gives at that background squeeze;
                       a joint asked for more collapses
  repulsion     0.1    acts only where joints carry charge
"""
from __future__ import annotations

from dataclasses import replace

from physics.rules.settings import RuleSettings

_ON = RuleSettings(energy_length=1.0, energy_flow=0.1, drift=0.0, attraction=3e-5,
                   joint_push=0.5, push_limit=0.01, repulsion=0.1, rewiring=0)


def all_rules_on(**overrides) -> RuleSettings:
    """RuleSettings with every rule on but rewiring and drift, plus the given overrides."""
    return replace(_ON, **overrides)
