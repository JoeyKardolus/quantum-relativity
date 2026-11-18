"""Search records: one probe of one size, and the outcome of a whole search.

`Probe`: `size` (cells a side), `fits` (False when the build raised MemoryError),
`cells` and `steps_per_second` (None when it did not fit), `facts` (the run's own facts,
read after its timing).
`Search`: `size`, `cells`, `steps_per_second` of the largest size that met the rate (all
None when not even the smallest did), `probes` in the order they ran, and `limit`, what
stopped the search going higher (limits.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Probe:
    """One size, built and timed, or refused for memory."""
    size: int
    fits: bool
    cells: int | None = None
    steps_per_second: float | None = None
    facts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Search:
    """The largest size that met the rate, every probe, and what stopped the search."""
    size: int | None
    cells: int | None
    steps_per_second: float | None
    probes: tuple
    limit: str
