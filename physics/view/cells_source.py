"""Cells source: the Planck-box engine (physics.potts) behind the viewer's source interface.

Contract (the source interface the `Stepper` reads; a test may supply any object
that has the same):
  reset()      back to a fresh start: `new_universe(cells_per_side,
               boxes_per_cell_side, seed, unevenness, slope)`, step 0
  step()       one engine step, `physics.potts.step(state, settings)`; works whether
               that step changes the state in place (returns None) or returns a new one
  steps        steps since the last reset
  view()       the occupied boxes, from `physics.potts.view_source(state)`: an object
               (or dict) with `positions` (n, 3) box centres in box units, `density`
               (n,) energy per box, `kind` (n,) ENERGY_VOLUME (0) or JOINT (1) for the
               cell owning the box, `border` (n,) nonzero on a cell border; Warp arrays
               on one device (`box_points.py` reads them)
  stats()      `physics.potts.stats(state, settings)`: a dict (keys the title reads are in
               `title_line.py`; a missing key shows as "?")
Settings are `Settings(**overrides)`; an unknown name raises TypeError at
construction. Everything stays on the GPU; the viewer gathers what it draws there
and copies only that.
"""
from __future__ import annotations

from types import SimpleNamespace

from physics.potts import Settings, new_universe, stats, step, view_source

_VIEW_FIELDS = ("positions", "density", "kind", "border")


class CellsSource:
    """A universe of cells stepped under all rules, read as its occupied Planck boxes."""

    def __init__(self, cells_per_side: int, boxes_per_cell_side: int, seed: int,
                 unevenness: float, slope: float, overrides: dict[str, float] | None = None) -> None:
        self.cells_per_side, self.boxes_per_cell_side = cells_per_side, boxes_per_cell_side
        self.seed, self.unevenness, self.slope = seed, unevenness, slope
        self.settings = Settings(**(overrides or {}))
        self.reset()

    def reset(self) -> None:
        """A fresh start from the same sizes, seed, unevenness and slope."""
        self.state = new_universe(self.cells_per_side, self.boxes_per_cell_side, self.seed,
                                  self.unevenness, self.slope)
        self.steps = 0

    def step(self) -> None:
        """One engine step, in place or returned."""
        stepped = step(self.state, self.settings)
        if stepped is not None:
            self.state = stepped
        self.steps += 1

    def view(self):
        """The occupied boxes: positions, density, owner kind and border flag."""
        found = view_source(self.state)
        if isinstance(found, dict):
            return SimpleNamespace(**{name: found[name] for name in _VIEW_FIELDS})
        return found

    def stats(self) -> dict:
        """The engine's counts for the title."""
        return stats(self.state, self.settings)
