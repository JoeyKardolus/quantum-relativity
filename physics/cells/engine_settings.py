"""EngineSettings: the strength of each cell rule; every rule is always on in a scene.

Contract: frozen record of floats, read by step() each step:
  attraction       pull a joint gets per unit of pull potential (rule 7): joint (a, b)
                   feels a squeeze force s * thickness, s = attraction * (p_a + p_b) / 2
  joint_push       strength of a joint's push toward its rest thickness R (rule 8): force
                   joint_push * R * ((R/t)^push_power - 1) / push_power on its rate at
                   thickness t (joint_push.py, swing_joints.py), slope joint_push at
                   rest; a push moves no energy. >= 0
  push_power       how steeply the push rises as a joint thins: > 0 grows without bound
                   as t goes to 0 (a hard wall), -1 is the straight spring
                   joint_push * (R - t), 0 is joint_push * R * ln(R/t)
  push_limit       a joint collapses when its thickness falls below push_limit times
                   its rest thickness; in [0, 1)
  joint_rest       a joint's rest thickness as a share of its two energy volumes' mean
                   length (volume ** (1/3)); see joint_rest.py
  leak             share of its energy every energy volume leaks along its seams each
                   step, split by its neighbours' pull (rule 6, leak.py); in [0, 1]
  seal_point       squeeze depth 1 - t / rest at which a seam closes (seam_openness.py);
                   a depth past 1 - push_limit acts as 1 - push_limit. >= 0
  split_threshold  an energy volume whose energy per measured cell volume falls below
                   this is stretched too thin and splits in two (split_cell.py);
                   0 never splits. Background energy 1 gives energy per volume 1.
`EngineSettings.all_on(**overrides)` gives the defaults below:
  attraction       0.005  even background (potential 10) squeezes joints by s = 0.05
  joint_push       0.1    with that squeeze a joint balances at 0.77 of rest
                          (joint_balance.py) and swings with period about
                          2 pi / sqrt(k (R/t)^4 + s) = 11 steps
  push_power       3.0    a round power, not tuned to a result: holding a joint at
                          the push limit takes s = 0.47, 9.3 times the background
                          squeeze (the straight spring needed 2 times); the swing's
                          stiffness there, k 2^4 + s = 2.1, keeps the step stable (< 4)
  push_limit       0.5    a joint below half its rest thickness gives way
  joint_rest       0.08   rest thickness 8% of the volumes' mean length: on a typical
                          Voronoi face (area about 0.375 V^(2/3)) the joint holds about
                          3% of an energy volume (rule 8: about 2 to 5%)
  leak             0.05   the rate the removed energy flow had, not tuned to a result
  seal_point       0.4    between the background balance depth (0.23) and the push
                          limit's (0.5); not tuned to a result
  split_threshold  0.05   below the lowest energy per volume of an unevenness-0.5 start
                          (0.5 ** 4 = 0.0625), so no volume starts torn; not tuned
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EngineSettings:
    """Strengths of attraction, joint push and its power, push limit, joint rest, leak, seal point, split threshold."""

    attraction: float
    joint_push: float
    push_power: float
    push_limit: float
    joint_rest: float
    leak: float
    seal_point: float
    split_threshold: float

    @classmethod
    def all_on(cls, **overrides: float) -> "EngineSettings":
        """Defaults with every rule on, plus the given overrides."""
        return replace(_ALL_ON, **overrides)


_ALL_ON = EngineSettings(attraction=0.005, joint_push=0.1, push_power=3.0, push_limit=0.5,
                         joint_rest=0.08, leak=0.05, seal_point=0.4, split_threshold=0.05)
