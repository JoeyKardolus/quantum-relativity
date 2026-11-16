"""Cells source: the whole-quanta engine (physics.engine) behind the viewer's source interface.

Contract (the source interface the `Stepper` reads; a test may supply any object
that has the same):
  reset()      back to a fresh start: `new_universe(cells_per_side,
               boxes_per_cell_side, seed, unevenness, slope,
               joint_unevenness=joint_unevenness)`, step 0
  step()       one engine step, `physics.engine.step(state, settings)`; works whether
               that step changes the state in place (returns None) or returns a new one
  steps        steps since the last reset
  view()       the occupied boxes, from `physics.engine.view_source(state)`: an object
               (or dict) with `positions` (n, 3) box centres in box units, `density`
               (n,) whole quanta per box ignoring sign, `kind` (n,) ENERGY_VOLUME (0, a
               plus box) or JOINT (1, a minus box), `border` (n,) nonzero where a face
               neighbour holds the other sign or no space; Warp arrays on one device
               (`box_points.py` reads them)
  stats()      `physics.engine.stats(state, settings)`: a dict (keys the title reads are in
               `title_line.py`; a missing key shows as "?")
Settings are `Settings(**overrides)`, a value cast to int where the field's default is an
int; an unknown name raises TypeError at construction. Everything stays on the GPU; the viewer gathers what it draws there
and copies only that.
"""
from __future__ import annotations

from types import SimpleNamespace

from physics.engine import Settings, new_universe, stats, step, view_source

_VIEW_FIELDS = ("positions", "density", "kind", "border")


class CellsSource:
    """A universe of cells stepped under all rules, read as its occupied Planck boxes."""

    def __init__(self, cells_per_side: int, boxes_per_cell_side: int, seed: int,
                 unevenness: float, slope: float, overrides: dict[str, float] | None = None,
                 joint_unevenness: float = 0.0) -> None:
        self.cells_per_side, self.boxes_per_cell_side = cells_per_side, boxes_per_cell_side
        self.seed, self.unevenness, self.slope = seed, unevenness, slope
        self.joint_unevenness = joint_unevenness
        self.settings = Settings(**_typed(overrides or {}))
        self.reset()

    def reset(self) -> None:
        """A fresh start from the same sizes, seed, both unevennesses and slope."""
        self.state = new_universe(self.cells_per_side, self.boxes_per_cell_side, self.seed,
                                  self.unevenness, self.slope, joint_unevenness=self.joint_unevenness)
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


def _typed(overrides: dict) -> dict:
    """The overrides with each value cast to the type of its field's default (int fields stay whole)."""
    defaults = Settings()
    return {name: type(getattr(defaults, name))(value) if hasattr(defaults, name) else value
            for name, value in overrides.items()}
