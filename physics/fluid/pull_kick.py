"""Pull kick: the pull's change to the A flow over one substep.

Contract: `pull_kick(grid, dt)` adds -a * grad(potential) * dt to a_momentum in
every box, a the A-energy, the gradient by central differences (one-sided in the
outermost boxes). The pull acts on A-energy only (bible: A pulls on A); B feels it
only through friction and the seal (rub.py). Moves no energy. In place.
"""
from __future__ import annotations

import warp as wp

from physics.fluid.grid import QUANTUM, Grid

_QUANTUM = wp.constant(QUANTUM)


@wp.func
def _slope(potential: wp.array3d(dtype=float), i: int, j: int, k: int, axis: int, side: int) -> float:
    low = wp.vec3i(i, j, k)
    high = wp.vec3i(i, j, k)
    if low[axis] > 0:
        low[axis] = low[axis] - 1
    if high[axis] < side - 1:
        high[axis] = high[axis] + 1
    width = float(high[axis] - low[axis])
    return (potential[high[0], high[1], high[2]] - potential[low[0], low[1], low[2]]) / width


@wp.kernel
def _kick(a_energy: wp.array3d(dtype=wp.int64), potential: wp.array3d(dtype=float),
          a_momentum: wp.array3d(dtype=wp.vec3), side: int, dt: float):
    i, j, k = wp.tid()
    energy = float(wp.max(a_energy[i, j, k], wp.int64(0))) * _QUANTUM
    if energy <= 0.0:
        return
    gradient = wp.vec3(_slope(potential, i, j, k, 0, side), _slope(potential, i, j, k, 1, side),
                       _slope(potential, i, j, k, 2, side))
    a_momentum[i, j, k] = a_momentum[i, j, k] - energy * dt * gradient


def pull_kick(grid: Grid, dt: float) -> None:
    """Accelerate the A flow down the pull potential for dt."""
    wp.launch(_kick, dim=(grid.side,) * 3, device=grid.a_energy.device,
              inputs=[grid.a_energy, grid.potential, grid.a_momentum, grid.side, float(dt)])
