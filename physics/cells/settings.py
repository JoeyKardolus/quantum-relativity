"""CellSettings: the strength of each cell rule; every rule is always on in a scene.

Contract: frozen record of floats, read by step():
  attraction   pull a joint gets per unit of pull potential (rule 7): each step it
               keeps exp(-attraction * mean potential of its two volumes) of its volume
  joint_push   share of the gap to its rest volume a joint swells back each step,
               taking that space from its two energy volumes (rule 8); in [0, 1]
  push_limit   a joint collapses when its volume falls below push_limit times its
               rest volume; in [0, 1)
  joint_rest   a joint's rest volume as a share of its two energy volumes' mean volume
  leak         share of its energy every energy volume leaks to its neighbours each
               step, split by their pull (rule 6, leak.py); in [0, 1]. With
               attraction 0 nothing pulls and nothing leaks.
`CellSettings.all_on(**overrides)` gives the defaults below:
  attraction   0.005  even background (potential 10) pulls each joint by 0.05 a step
  joint_push   0.1    with that pull a joint settles near 0.7 of its rest volume
                      (k / (1 - (1 - k) exp(-pull)), k = joint_push)
  push_limit   0.5    a joint pulled about 2.4 times as hard as the background
                      settles below half its rest volume and gives way
  joint_rest   0.2    joints hold about a sixth of space at rest (2 joints per volume)
  leak         0.05   the rate the removed energy flow had, not tuned to a result
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CellSettings:
    """Strengths of attraction, joint push, push limit, joint rest and leak."""

    attraction: float
    joint_push: float
    push_limit: float
    joint_rest: float
    leak: float

    @classmethod
    def all_on(cls, **overrides: float) -> "CellSettings":
        """Defaults with every rule on, plus the given overrides."""
        return replace(_ALL_ON, **overrides)


_ALL_ON = CellSettings(attraction=0.005, joint_push=0.1, push_limit=0.5, joint_rest=0.2,
                       leak=0.05)
