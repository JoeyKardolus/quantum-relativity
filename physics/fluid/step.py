"""Step: advance the energy fluid by settings.step_time, in place.

Contract: `step(grid, settings)` changes grid in place and returns None. It runs
substeps until step_time has passed; each substep:
  1. pull_potential: settings.potential_cycles V-cycles, warm-started
  2. dt = min(courant / fastest_wave, time left) (CFL from the fastest wave)
  3. pull_kick: the pull accelerates A's flow
  4. push_kick: B's pressure accelerates B's flow
  5. rub: A-through-B friction, the seal lock
  6. transport: energy and momentum cross the faces by that flow (exact integer energy)
First-order operator splitting in that order. grid.steps grows by 1, grid.substeps
by the substeps taken. Raises ValueError when a flow velocity stops being finite
(fastest_wave.py) or one step would need more than settings.max_substeps; the grid
is then left as the failing substep found it, partly stepped.
"""
from __future__ import annotations

from physics.fluid.fastest_wave import fastest_wave
from physics.fluid.grid import Grid
from physics.fluid.pull_kick import pull_kick
from physics.fluid.push_kick import push_kick
from physics.fluid.pull_potential import pull_potential
from physics.fluid.rub import rub
from physics.fluid.settings import Settings
from physics.fluid.transport import transport


def step(grid: Grid, settings: Settings) -> None:
    """One step of step_time, in CFL substeps."""
    remaining = settings.step_time
    taken = 0
    while remaining > 1e-9 * settings.step_time:
        if taken >= settings.max_substeps:
            raise ValueError(f"step {grid.steps + 1}: more than {int(settings.max_substeps)} substeps needed")
        pull_potential(grid, settings, int(settings.potential_cycles))
        speed = fastest_wave(grid, settings)
        dt = remaining if speed <= 0.0 else min(remaining, settings.courant / speed)
        pull_kick(grid, dt)
        push_kick(grid, settings, dt)
        rub(grid, settings, dt)
        transport(grid, dt)
        remaining -= dt
        taken += 1
    grid.steps += 1
    grid.substeps += taken
