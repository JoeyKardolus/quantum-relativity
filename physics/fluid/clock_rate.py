"""Clock rate: how fast each box's own time runs, set by its energy concentration (local time).

Contract: `clock_rate(grid, settings)` writes grid.clock in place: per box
  q    = (a / a_rest + b / settings.b_rest) / 2,   a_rest = grid.start_a_mean
  rate = (s + 1) / (s + q),                        s = settings.clock_scale
a, b the A- and B-energy in the box. q is the box's energy concentration with the
joints counted as much as the energy volumes, each kind against its own rest (bible
foundation 4: "each region's clock runs at a rate set by its own energy
concentration, joints included"), so a box whose joints the pull squeezes runs slow
even when it holds little A. The start's mean box (q = 1) runs at 1, a more packed
box slower (rate -> 0 as q grows), an empty box fastest, (s + 1) / s. The step is
only the computer's counter: every process in a box (pull and push kicks, friction,
the leak, what it sends across its faces) runs for rate * dt of each substep's dt
(step.py). Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.settings import Settings

_QUANTUM = wp.constant(QUANTUM)


@wp.kernel
def _rate(a_energy: wp.array3d(dtype=wp.int64), b_energy: wp.array3d(dtype=wp.int64), a_rest: float,
          b_rest: float, scale: float, clock: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    pulling = float(wp.max(a_energy[i, j, k], wp.int64(0))) * _QUANTUM
    pushing = float(wp.max(b_energy[i, j, k], wp.int64(0))) * _QUANTUM
    concentration = 0.5 * (pulling / a_rest + pushing / b_rest)
    clock[i, j, k] = (scale + 1.0) / (scale + concentration)


def clock_rate(grid: Grid, settings: Settings) -> None:
    """Set every box's clock rate from its energy concentration, joints included."""
    wp.launch(_rate, dim=(grid.side,) * 3, device=grid.a_energy.device,
              inputs=[grid.a_energy, grid.b_energy, grid.start_a_mean, settings.b_rest, settings.clock_scale,
                      grid.clock])
