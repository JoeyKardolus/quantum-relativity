"""Rates: what a timing says, and the formula that turns a count and a time into it.

`rates_law(cells, steps, seconds)` returns a `Rates`:
  steps_per_second       steps / seconds
  cell_steps_per_second  cells * steps / seconds
with `cells`, `steps` and `seconds` kept beside them. Raises ValueError when seconds is
not above 0 (a timing that took no time has no rate).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rates:
    """One timing: the counts, the wall time and the two rates."""
    cells: int
    steps: int
    seconds: float
    steps_per_second: float
    cell_steps_per_second: float


def rates_law(cells: int, steps: int, seconds: float) -> Rates:
    """The rates of `steps` steps of a `cells`-cell universe that took `seconds`."""
    if seconds <= 0:
        raise ValueError(f"a timing needs seconds above 0, got {seconds}")
    return Rates(cells, steps, seconds, steps / seconds, cells * steps / seconds)
