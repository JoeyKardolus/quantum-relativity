"""Rules layer: the connection-theory rules that step a network forward.

Module map:
  settings.py  RuleSettings: per-rule strengths, 0 = off
  step.py      step: energy flow, drift, attraction, repulsion, energy-length, rewiring
  rewire.py    rewire: local connection slides accepted when they lower the action
"""
from physics.rules.settings import RuleSettings
from physics.rules.rewire import rewire
from physics.rules.step import step

__all__ = ["RuleSettings", "step", "rewire"]
