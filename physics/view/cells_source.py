"""Cells source: the engine (physics.engine) behind the viewer's source interface.

Contract (the source interface the `Stepper` and `saved_frames` read; a test may supply
any object that has the same):
  reset()      back to a fresh start: `new_universe(cells_per_side, seed, unevenness,
               slope)`, step 0
  step()       one engine step, `physics.engine.step(state, settings)`
  steps        steps since the last reset
  view()       the grid now as a Pieces record (`cell_pieces`), laid out from the
               engine's kinds, lengths, packings and its record of tears: one host copy
               of three per-cell arrays per call, no box is read
  stats()      `physics.engine.stats(state, settings)`: a dict (keys the title reads are
               in `title_line.py`; a missing key shows as "?")
Settings are `Settings(**overrides)`; an unknown name raises TypeError at construction.
"""
from __future__ import annotations

from physics.engine import Settings, new_universe, stats, step
from physics.view.cell_pieces import Pieces, cell_pieces


class CellsSource:
    """A universe of cells stepped under all rules, read as the blocks of its warped grid."""

    def __init__(self, cells_per_side: int = 32, seed: int = 1, unevenness: float = 0.1, slope: float = -3.0,
                 overrides: dict[str, float] | None = None) -> None:
        self.cells_per_side, self.seed, self.unevenness, self.slope = cells_per_side, seed, unevenness, slope
        self.settings = Settings(**(overrides or {}))
        self.reset()

    def reset(self) -> None:
        """A fresh start from the same size, seed, unevenness and slope."""
        self.state = new_universe(self.cells_per_side, self.seed, self.unevenness, self.slope)
        self.steps = 0

    def step(self) -> None:
        """One engine step."""
        step(self.state, self.settings)
        self.steps += 1

    def view(self) -> Pieces:
        """Every living cell's block in the warped grid."""
        state = self.state
        return cell_pieces(state.side, state.kind_h[:state.count], state.host("length"), state.host("packing"),
                           state.torn_from)

    def stats(self) -> dict:
        """The engine's counts for the title."""
        return stats(self.state, self.settings)
