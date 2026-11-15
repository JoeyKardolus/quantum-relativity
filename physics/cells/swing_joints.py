"""Swing joints: one step of every joint's swing, a spring with no damping (bible rule 5).

Contract: `swing_joints(thickness, rate, rest, squeeze, push, power)` returns
(thickness, rate) after one step, per joint:
  force      joint_push(thickness, rest, push, power) - squeeze * thickness
             (its push toward rest, steep as the joint thins, rule 8 and
             joint_push.py, against the pull that squeezes it,
             squeeze = attraction * (p_a + p_b) / 2, rule 7)
  stiffness  the force's slope, push * (rest/thickness)^(power+1) + squeeze
  numerics   velocity Verlet (symplectic: no numerical damping or growth) over
             the step cut into equal substeps, as many as keep every joint's
             swing angle per substep, sqrt(stiffness) / substeps, at or below
             STEP_ANGLE (0.25 rad; Verlet goes unstable at 2). The stiffness is read
             at each joint's thinnest turning point, where its swing energy
             rate^2 / 2 + V(thickness) meets the convex potential V = -integral of
             the force (found by TURNING_BISECTIONS halvings): the stiffest point
             of the whole swing, the same every step while the energy holds, so a
             lone joint keeps one substep width and Verlet keeps its swing. At
             most MAX_SUBSTEPS (4096), and a swing that reaches a hard wall's floor
             (push > 0, power > 0, MIN_SHARE of rest) raises RuntimeError rather
             than run unstable.
rate is thickness change per step, as before. Rest and squeeze are held over the
step. So a joint overshoots its balance (joint_balance.py) and keeps swinging around
it; nothing damps it. Push moves no energy: no energy volume is read or changed
here. step.py collapses a joint below push_limit times rest.
"""
from __future__ import annotations

import math

import numpy as np

from physics.cells.joint_push import MIN_SHARE, joint_push

STEP_ANGLE = 0.25
MAX_SUBSTEPS = 4096
TURNING_BISECTIONS = 40


def swing_joints(thickness: np.ndarray, rate: np.ndarray, rest: np.ndarray, squeeze: np.ndarray,
                 push: float, power: float) -> tuple[np.ndarray, np.ndarray]:
    """Thickness and rate of every joint after one undamped step, velocity Verlet in substeps."""
    if len(thickness) == 0:
        return thickness, rate
    substeps = _substeps(thickness, rate, rest, squeeze, push, power)
    width = 1.0 / substeps
    accel = _force(thickness, rest, squeeze, push, power)
    for _ in range(substeps):
        thickness = thickness + width * rate + 0.5 * width * width * accel
        next_accel = _force(thickness, rest, squeeze, push, power)
        rate = rate + 0.5 * width * (accel + next_accel)
        accel = next_accel
    return thickness, rate


def _force(thickness: np.ndarray, rest: np.ndarray, squeeze: np.ndarray, push: float,
           power: float) -> np.ndarray:
    """Per joint, its push toward rest minus the squeeze on it."""
    return joint_push(thickness, rest, push, power) - squeeze * thickness


def _substeps(thickness: np.ndarray, rate: np.ndarray, rest: np.ndarray, squeeze: np.ndarray,
              push: float, power: float) -> int:
    """Substeps that keep the stiffest joint's swing at or below STEP_ANGLE per substep."""
    floor = MIN_SHARE * rest
    thinnest = _turning_point(thickness, rate, rest, squeeze, push, power)
    walled = push > 0 and power > 0
    if walled and np.any((thinnest <= floor * (1 + 1e-9)) & (rest > 0)):
        raise RuntimeError("joint swing: a joint swings into its push's floor (MIN_SHARE of rest)")
    ratio = np.divide(rest, thinnest, out=np.zeros_like(thinnest, dtype=float), where=thinnest > 0)
    stiffest = float(np.max(push * ratio ** (power + 1) + np.abs(squeeze)))
    if not math.isfinite(stiffest):
        raise RuntimeError("joint swing: a joint's stiffness is not finite")
    needed = max(1, math.ceil(math.sqrt(stiffest) / STEP_ANGLE))
    if needed > MAX_SUBSTEPS:
        raise RuntimeError(f"joint swing needs {needed} substeps (stiffness {stiffest:.3g}), "
                           f"more than {MAX_SUBSTEPS}")
    return needed


def _turning_point(thickness: np.ndarray, rate: np.ndarray, rest: np.ndarray, squeeze: np.ndarray,
                   push: float, power: float) -> np.ndarray:
    """Per joint, the thinnest thickness its swing energy reaches (bisection on the convex potential)."""
    floor = np.maximum(MIN_SHARE * rest, 0.0)
    level = 0.5 * rate * rate + _potential(thickness, rest, squeeze, push, power)
    low, high = floor.copy(), np.maximum(thickness, floor)
    for _ in range(TURNING_BISECTIONS):
        middle = (low + high) / 2
        inside = _potential(middle, rest, squeeze, push, power) <= level
        low, high = np.where(inside, low, middle), np.where(inside, middle, high)
    return low


def _potential(thickness: np.ndarray, rest: np.ndarray, squeeze: np.ndarray, push: float,
               power: float) -> np.ndarray:
    """Per joint, minus the integral of the force: the swing's potential energy (up to a constant)."""
    safe_rest = np.where(rest > 0, rest, 1.0)
    held = np.maximum(thickness, MIN_SHARE * safe_rest)
    if power == 0:
        pushed = push * safe_rest * (held * np.log(safe_rest / held) + held)
    elif power == 1:
        pushed = push * safe_rest * (safe_rest * np.log(held) - held)
    else:
        pushed = push * safe_rest / power * (safe_rest ** power * held ** (1 - power) / (1 - power)
                                             - held)
    return np.where(rest > 0, -pushed, 0.0) + 0.5 * squeeze * thickness * thickness
