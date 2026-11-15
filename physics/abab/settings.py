"""Settings: the strength of each ABAB rule; every rule is always on in a scene.

Contract: frozen record of floats, read by step() each step:
  attraction   joint b feels a squeeze s = attraction * (mean pull potential of the
               energy volumes it touches) (pull_potential.py, bible rule 7)
  joint_push   strength of a joint's push toward its rest size R (joint_rest_size.py):
               joint_push * R * ((R/t)^push_power - 1) / push_power at size t
               (physics.abab.joint_push); moves no energy. >= 0
  push_power   how steeply the push rises as a joint is squeezed; > 0 is a hard wall
  push_limit   a joint smaller than push_limit times its rest size collapses; in [0, 1)
  leak         share of its energy an energy volume gives each step through its
               open joints, split by pull (leak.py); in [0, 1]
  seal_point   squeeze depth 1 - t / R at which a joint passes nothing
               (joint_openness.py); a depth past 1 - push_limit acts as 1 - push_limit
  split_size   an energy volume larger than split_size times the start's mean energy
               volume splits in two (split_volumes.py); > 0, inf never splits
`Settings.all_on(**overrides)` gives the defaults: the cells engine's values
(the removed physics/cells), carried over unchanged and not tuned here:
attraction 0.005, joint_push 0.1, push_power 3, push_limit 0.5, leak 0.05,
seal_point 0.4, split_size 4.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    """Strengths of attraction, joint push and its power, push limit, leak, seal point, split size."""

    attraction: float
    joint_push: float
    push_power: float
    push_limit: float
    leak: float
    seal_point: float
    split_size: float

    @classmethod
    def all_on(cls, **overrides: float) -> "Settings":
        """Defaults with every rule on, plus the given overrides."""
        return replace(_ALL_ON, **overrides)


_ALL_ON = Settings(attraction=0.005, joint_push=0.1, push_power=3.0, push_limit=0.5,
                   leak=0.05, seal_point=0.4, split_size=4.0)
