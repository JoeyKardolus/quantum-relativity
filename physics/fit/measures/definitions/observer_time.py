"""Observer time: the time that passes on the clock where the energy sits.

Contract: `observer_time(counter_time, rate_before, rate_now)` returns the observer's
time between two samples: the counter time between them times the mean of the two
samples' observer rates (the energy volumes' clock rates weighted by their quanta). Time
is local: an astronomer is made of energy, so the fit times the expansion on that clock.
"""
from __future__ import annotations


def observer_time(counter_time: float, rate_before: float, rate_now: float) -> float:
    """Counter time times the mean of the two observer rates."""
    return counter_time * 0.5 * (rate_before + rate_now)
