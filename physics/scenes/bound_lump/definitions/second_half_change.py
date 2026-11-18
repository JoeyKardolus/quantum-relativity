"""Second half change: by how much a series changes over the second half of its samples.

Contract: `second_half_change(values)` takes a sequence of floats in time order and
returns the last one over the one at index len // 2, a float: 1 for no change. Raises
ValueError for an empty series or a middle value of 0.
"""
from __future__ import annotations


def second_half_change(values) -> float:
    """The last sample over the middle sample."""
    values = list(values)
    if not values or values[len(values) // 2] == 0:
        raise ValueError("need samples, and a middle sample that is not 0")
    return float(values[-1] / values[len(values) // 2])
