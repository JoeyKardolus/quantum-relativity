"""Source interface: what the stepper and the saved pictures ask of whatever they draw.

Contract: `Source` is a Protocol; the engine source (`CellsSource`) and any test double
that has the same five names fit it
(`isinstance(thing, Source)` asks exactly that).
  reset()   back to a fresh start, step 0
  step()    one step
  steps     steps since the last reset
  view()    the grid now as a `Pieces` record
  stats()   a dict of counts for the title; its names are in stat_names.py, and a
            missing one shows as "?"
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from physics.view.layout.definitions.pieces import Pieces


@runtime_checkable
class Source(Protocol):
    """Something that steps and can be drawn."""

    steps: int

    def reset(self) -> None:
        """Back to a fresh start."""

    def step(self) -> None:
        """One step."""

    def view(self) -> Pieces:
        """The grid now."""

    def stats(self) -> dict:
        """Counts for the title."""
