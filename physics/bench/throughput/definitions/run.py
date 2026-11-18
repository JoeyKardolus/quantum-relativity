"""Run: one universe ready to be timed.

`advance()` takes the universe one step. `cells` is its cell count at the start (cells a
side, cubed; tears and collapses change the count during a run, the rate is quoted on
the start). `facts()` returns whatever else the builder can say about it once it has
run (a dict, empty by default); it is read after the timing, never inside it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Run:
    """A universe to time: its one-step callable, its start cell count and its facts."""
    advance: Callable[[], None]
    cells: int
    facts: Callable[[], dict] = dict
