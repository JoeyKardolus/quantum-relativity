"""Torn out: a universe that tears without end.

Contract: `torn_out(tears, start_volumes)` returns True when there were more than
`TEARS_PER_VOLUME` (2) tears per energy volume of the start. It is read from the grid's
tear count after every step, so a tear chain is caught between samples.
"""
from __future__ import annotations

TEARS_PER_VOLUME = 2


def torn_out(tears: int, start_volumes: int) -> bool:
    """Whether there were more than TEARS_PER_VOLUME tears per start volume."""
    return tears > TEARS_PER_VOLUME * start_volumes
