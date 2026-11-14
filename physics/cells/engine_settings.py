"""EngineSettings: the strength of each cell rule; every rule is always on in a scene.

Contract: frozen record of floats, read by apply_rules() (step.py runs it each step):
  attraction   pull a joint gets per unit of pull potential (rule 7): each step it
               keeps exp(-attraction * mean potential of its two volumes) of its volume
  joint_push   share of the gap to its rest volume a joint swells back each step
               (rule 8); the swelling is new space, it takes nothing from the energy
               volumes and moves no energy; in [0, 1]
  push_limit   a joint collapses when its volume falls below push_limit times its
               rest volume; in [0, 1)
  joint_rest   a joint's rest volume as a share of its two energy volumes' mean volume
  leak         share of its energy every energy volume leaks along its seams each
               step, split by its neighbours' pull (rule 6, leak.py); in [0, 1].
               With attraction 0 nothing pulls and nothing leaks.
  seal_point   squeeze depth 1 - J / R at which a seam closes (seam_openness.py):
               its openness falls in a straight line from 1 at rest to 0 here; a
               depth reached before the push limit's 1 - push_limit (a larger value
               acts as 1 - push_limit). >= 0.
`EngineSettings.all_on(**overrides)` gives the defaults below:
  attraction   0.005  even background (potential 10) pulls each joint by 0.05 a step
  joint_push   0.1    with that pull a joint settles near 0.7 of its rest volume
                      (k / (1 - (1 - k) exp(-pull)), k = joint_push)
  push_limit   0.5    a joint pulled about 2.4 times as hard as the background
                      settles below half its rest volume and gives way
  joint_rest   0.03   a joint at rest is 3% of its two energy volumes' mean volume,
                      a thin wall (rule 8: about 2 to 5%)
  leak         0.05   the rate the removed energy flow had, not tuned to a result
  seal_point   0.4    halfway from the background joint (about 0.7 of rest, depth
                      0.3) to the push limit (depth 0.5): background seams stay a
                      quarter open, a joint squeezed past it seals; not tuned to a result
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EngineSettings:
    """Strengths of attraction, joint push, push limit, joint rest, leak and seal point."""

    attraction: float
    joint_push: float
    push_limit: float
    joint_rest: float
    leak: float
    seal_point: float

    @classmethod
    def all_on(cls, **overrides: float) -> "EngineSettings":
        """Defaults with every rule on, plus the given overrides."""
        return replace(_ALL_ON, **overrides)


_ALL_ON = EngineSettings(attraction=0.005, joint_push=0.1, push_limit=0.5, joint_rest=0.03,
                       leak=0.05, seal_point=0.4)
