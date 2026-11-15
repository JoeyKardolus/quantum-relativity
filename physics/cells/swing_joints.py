"""Swing joints: one step of every joint's swing, a spring with no damping (bible rule 5).

Contract: `swing_joints(thickness, rate, rest, squeeze, push)` returns (thickness, rate)
after one step, per joint:
  force      push * (rest - thickness) - squeeze * thickness
             (its push toward rest, rule 8, against the pull that squeezes it,
             squeeze = attraction * (p_a + p_b) / 2, rule 7)
  rate      += force          (the joint's momentum)
  thickness += rate (new)     (semi-implicit Euler: no numerical damping or growth
                               for a steady spring while sqrt(push + squeeze) < 2)
So a joint overshoots its balance push * rest / (push + squeeze) and keeps swinging
around it; nothing damps it. Push moves no energy: no energy volume is read or
changed here. A thickness is never set below zero here; step.py collapses a joint
long before (push_limit times rest).
"""
from __future__ import annotations

import numpy as np


def swing_joints(thickness: np.ndarray, rate: np.ndarray, rest: np.ndarray, squeeze: np.ndarray,
                 push: float) -> tuple[np.ndarray, np.ndarray]:
    """Thickness and rate of every joint after one undamped spring step."""
    new_rate = rate + push * (rest - thickness) - squeeze * thickness
    return thickness + new_rate, new_rate
