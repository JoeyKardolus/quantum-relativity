"""Swing joints: one step of every joint's swing, a spring with no damping (bible rule 5).

Contract: `swing_joints(thickness, rate, rest, squeeze, push, power)` returns
(thickness, rate) after one step, per joint:
  force      joint_push(thickness, rest, push, power) - squeeze * thickness
             (its push toward rest, steep as the joint thins, rule 8 and
             joint_push.py, against the pull that squeezes it,
             squeeze = attraction * (p_a + p_b) / 2, rule 7)
  rate      += force          (the joint's momentum)
  thickness += rate (new)     (semi-implicit Euler: no numerical damping or growth
                               while the spring's stiffness, the force's slope
                               push * (rest/thickness)^(power+1) + squeeze, stays
                               below 4)
So a joint overshoots its balance (joint_balance.py) and keeps swinging around it;
nothing damps it. Push moves no energy: no energy volume is read or changed here.
step.py collapses a joint below push_limit times rest.
"""
from __future__ import annotations

import numpy as np

from physics.cells.joint_push import joint_push


def swing_joints(thickness: np.ndarray, rate: np.ndarray, rest: np.ndarray, squeeze: np.ndarray,
                 push: float, power: float) -> tuple[np.ndarray, np.ndarray]:
    """Thickness and rate of every joint after one undamped spring step."""
    new_rate = rate + joint_push(thickness, rest, push, power) - squeeze * thickness
    return thickness + new_rate, new_rate
