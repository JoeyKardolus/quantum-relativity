"""Joined phase: the phase of two fluids that come together in one box.

Contract: `joined_phase_host(quanta_a, phase_a, quanta_b, phase_b)` (host numpy; arrays
of one shape, the quanta counted ignoring sign) returns the phase, in (-pi, pi], of
each box that holds both: the angle of `quanta_a e^{i phase_a} + quanta_b e^{i phase_b}`.
Each fluid counts by its quanta, so the joined fluid moves with what the two brought:
a few quanta arriving do not turn a full box, and no motion is made by the joining.
Waves add and cancel (axiom 7): two fluids of equal quanta and opposite phase cancel,
and where the sum is under a million millionth of the two the box keeps `phase_a`, so
rounding never hands it an angle of its own. Cancelling takes direction and never
quanta: the amounts are the caller's, added as whole numbers.
`arrival_law` (axiom 4) is the same rule in a kernel.
"""
from __future__ import annotations

import numpy as np

CANCELLED = 1e-12


def joined_phase_host(quanta_a: np.ndarray, phase_a: np.ndarray, quanta_b: np.ndarray,
                      phase_b: np.ndarray) -> np.ndarray:
    """The phase of every box after fluid b has joined fluid a in it; a's own where the two cancel."""
    first = np.abs(np.asarray(quanta_a)).astype(np.float64)
    second = np.abs(np.asarray(quanta_b)).astype(np.float64)
    own = np.asarray(phase_a, dtype=np.float64)
    joined = first * np.exp(1j * own) + second * np.exp(1j * np.asarray(phase_b, dtype=np.float64))
    angle = np.angle(joined)
    angle = np.where(angle <= -np.pi, np.pi, angle)
    return np.where(np.abs(joined) <= CANCELLED * (first + second), own, angle)
