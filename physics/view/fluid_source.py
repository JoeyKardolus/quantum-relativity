"""Fluid source: physics.fluid's grid behind the viewer's source interface.

Contract (the source interface the `Stepper` reads; a test may supply any object
that has the same):
  reset()      back to a fresh start: `new_grid(side, seed, unevenness, slope)`, step 0
  step()       one engine step, `physics.fluid.step(grid, settings)`; works whether
               that step changes the grid in place (returns None) or returns a new one
  steps        steps since the last reset
  side         boxes along each edge (side^3 boxes)
  a_energy     the grid's per-box pulling (A) energy, a Warp array (side^3 boxes)
  b_energy     the grid's per-box pushing (B) energy, same shape and device
  stats()      `physics.fluid.stats(grid)`: a dict (keys the title reads:
               total_energy, size_vs_start, lumps, sealed_lumps, collapsed_cores,
               max_density; a missing key shows as "?")
Settings are `Settings.all_on(**overrides)`; an unknown name raises TypeError at
construction. The energies stay on the GPU; the viewer gathers what it draws there
(`box_points.py`) and copies only that.
"""
from __future__ import annotations

from physics.fluid import Settings, new_grid, stats, step
from physics.fluid.new_grid import SLOPE


class FluidSource:
    """A fluid grid stepped under all rules on, read as per-box A and B energy."""

    def __init__(self, side: int, seed: int, unevenness: float,
                 overrides: dict[str, float] | None = None, slope: float = SLOPE) -> None:
        self.side, self.seed, self.unevenness, self.slope = side, seed, unevenness, slope
        self.settings = Settings.all_on(**(overrides or {}))
        self.reset()

    def reset(self) -> None:
        """A fresh start from the same side, seed, unevenness and slope."""
        self.grid = new_grid(self.side, self.seed, self.unevenness, self.slope)
        self.steps = 0

    def step(self) -> None:
        """One engine step, in place or returned."""
        stepped = step(self.grid, self.settings)
        if stepped is not None:
            self.grid = stepped
        self.steps += 1

    @property
    def a_energy(self):
        """Pulling energy per box."""
        return self.grid.a_energy

    @property
    def b_energy(self):
        """Pushing energy per box."""
        return self.grid.b_energy

    def stats(self) -> dict:
        """The engine's counts for the title."""
        return stats(self.grid)
