"""Rules layer: the quantum-relativity theory rules that step a network forward.

Module map:
  settings.py  RuleSettings: per-rule strengths, 0 = off
  step.py      step: energy flow, drift, attraction, joint push, repulsion,
               energy-length, joint collapse, rewiring
  pull.py      pull: per-joint pull from energy volume, spread through the connections
  push.py      push: per-joint push back, proportional to squeeze, capped at push_limit
  collapse.py  collapse: joints that gave way merge into a neighbour, keeping all energy
  rewire.py    rewire: local connection slides accepted when they lower the action
"""
from physics.rules.settings import RuleSettings
from physics.rules.collapse import collapse
from physics.rules.pull import pull
from physics.rules.push import push
from physics.rules.rewire import rewire
from physics.rules.step import step

__all__ = ["RuleSettings", "step", "pull", "push", "collapse", "rewire"]
