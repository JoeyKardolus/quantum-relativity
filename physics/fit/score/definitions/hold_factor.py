"""Hold factor: how much of the universe still stands at the end of a run.

Contract: `hold_factor(alive, spent_share)` returns `alive * (1 - spent_share)`, each
clipped to 0 .. 1 first. `alive` is the share of the start's energy volumes still there,
so a universe that fell into a single cell gets about 0; `spent_share` is the share of
the size in overstretched volumes, so a universe that tears without end gets about 0.
Both fall off in a straight line: there is no cliff, and a collapse that makes one black
hole costs that one volume's share.
"""
from __future__ import annotations


def _share(value: float) -> float:
    """A value clipped to 0 .. 1."""
    return min(max(value, 0.0), 1.0)


def hold_factor(alive: float, spent_share: float) -> float:
    """alive * (1 - spent_share), each clipped to a share."""
    return _share(alive) * (1.0 - _share(spent_share))
