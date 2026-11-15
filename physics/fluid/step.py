"""Step: advance the energy fluid by settings.step_time, in place, every box in its own time.

Contract: `step(grid, settings)` changes grid in place and returns None. It runs
substeps until step_time has passed on the computer's counter; each substep:
  1. pull_potential: settings.potential_cycles V-cycles, warm-started
  2. clock_rate: each box's clock rate from its energy concentration, joints included
  3. dt = min(courant / fastest_wave, time left) (CFL, each box's waves in its own time)
  4. pull_kick: the pull accelerates A's flow
  5. push_kick: B's pressure accelerates B's flow
  6. rub: A-through-B friction (its loss becomes heat), the seal lock
  7. leak: A-energy through open joints to the neighbours, split by their pull (drift)
  8. transport: energy, momentum and heat cross the faces by that flow (exact integer energy)
Steps 4 to 8 run each box for its own time, clock * dt (bible: time is local; the
step is only the counter). A packed box so advances less per step, and its waves set
the substep only as fast as its clock lets them: substeps are counted in each
region's own time. First-order operator splitting in that order. grid.steps grows by
1, grid.substeps by the substeps taken. Raises ValueError when a flow velocity stops
being finite (fastest_wave.py) or one step would need more than
settings.max_substeps; the grid is then left as the failing substep found it,
partly stepped.
"""
from __future__ import annotations

from physics.fluid.clock_rate import clock_rate
from physics.fluid.fastest_wave import fastest_wave
from physics.fluid.grid import Grid
from physics.fluid.leak import leak
from physics.fluid.pull_kick import pull_kick
from physics.fluid.push_kick import push_kick
from physics.fluid.pull_potential import pull_potential
from physics.fluid.rub import rub
from physics.fluid.settings import Settings
from physics.fluid.transport import transport


def step(grid: Grid, settings: Settings) -> None:
    """One step of step_time, in CFL substeps, each box in its own time."""
    remaining = settings.step_time
    taken = 0
    while remaining > 1e-9 * settings.step_time:
        if taken >= settings.max_substeps:
            raise ValueError(f"step {grid.steps + 1}: more than {int(settings.max_substeps)} substeps needed")
        pull_potential(grid, settings, int(settings.potential_cycles))
        clock_rate(grid, settings)
        speed = fastest_wave(grid, settings)
        dt = remaining if speed <= 0.0 else min(remaining, settings.courant / speed)
        pull_kick(grid, dt)
        push_kick(grid, settings, dt)
        rub(grid, settings, dt)
        leak(grid, settings, dt)
        transport(grid, settings, dt)
        remaining -= dt
        taken += 1
    grid.steps += 1
    grid.substeps += taken
