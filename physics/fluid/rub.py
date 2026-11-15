"""Rub: friction where A-energy passes through B, and the seal that stops it passing.

Contract: `rub(grid, settings, dt)` changes the two momenta of every box holding both
kinds (each above the velocity floor, flow_velocity.py), in place, never the energy
and never the box's total momentum. With compression c = b / settings.b_rest:
  sealed                 the box, or one of its 6 face neighbours, has c at or past
                         seal_compression: A cannot pass through B there, so both kinds
                         take their common velocity (pa + pb) / (a + b); both sides of
                         every face that touches a sealed box are locked, so no A
                         crosses it except inside its B (transport.py): a packed lump
                         keeps its A
  otherwise              the drag rate on A's flow relative to B is friction * c (more
                         compressed B rubs harder; zero where there is no B), applied
                         exactly over dt: the relative velocity w = u_A - u_B decays as
                         w * exp(-friction * c * (1 + a / b) * dt), the common velocity
                         unchanged (unconditionally stable, momentum conserved)
Runs after the kicks and before transport, so the flow transport moves by is the
rubbed one. There is no other friction anywhere. The kinetic energy friction takes from directed
flow is heat; this engine does not keep it (total energy here is the A- and
B-energy in the boxes, which friction never changes).
"""
from __future__ import annotations

import warp as wp

from physics.fluid.flow_velocity import VELOCITY_FLOOR
from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.settings import Settings

_QUANTUM = wp.constant(QUANTUM)
_FLOOR = wp.constant(VELOCITY_FLOOR)


@wp.func
def _sealed_near(b_energy: wp.array3d(dtype=wp.int64), i: int, j: int, k: int, side: int,
                 seal_quanta: float) -> bool:
    """The box or one of its face neighbours holds B at or past the seal."""
    if float(b_energy[i, j, k]) >= seal_quanta:
        return True
    for axis in range(3):
        for towards in range(2):
            other = wp.vec3i(i, j, k)
            other[axis] = other[axis] + 2 * towards - 1
            if other[axis] >= 0 and other[axis] < side:
                if float(b_energy[other[0], other[1], other[2]]) >= seal_quanta:
                    return True
    return False


@wp.kernel
def _rub(a_energy: wp.array3d(dtype=wp.int64), b_energy: wp.array3d(dtype=wp.int64),
         a_momentum: wp.array3d(dtype=wp.vec3), b_momentum: wp.array3d(dtype=wp.vec3),
         side: int, b_rest: float, seal: float, friction: float, dt: float):
    i, j, k = wp.tid()
    pulling = float(a_energy[i, j, k]) * _QUANTUM
    pushing = float(b_energy[i, j, k]) * _QUANTUM
    if pulling <= _FLOOR or pushing <= _FLOOR:
        return
    total = pulling + pushing
    common = (a_momentum[i, j, k] + b_momentum[i, j, k]) / total
    compression = pushing / b_rest
    relative = wp.vec3(0.0, 0.0, 0.0)
    if not _sealed_near(b_energy, i, j, k, side, seal * b_rest / _QUANTUM):
        relative = (a_momentum[i, j, k] / pulling - b_momentum[i, j, k] / pushing) * wp.exp(
            -friction * compression * (1.0 + pulling / pushing) * dt)
    a_momentum[i, j, k] = pulling * (common + relative * (pushing / total))
    b_momentum[i, j, k] = pushing * (common - relative * (pulling / total))


def rub(grid: Grid, settings: Settings, dt: float) -> None:
    """Drag A's flow towards B's by the B compression, or lock them where B is sealed."""
    wp.launch(_rub, dim=(grid.side,) * 3, device=grid.a_energy.device,
              inputs=[grid.a_energy, grid.b_energy, grid.a_momentum, grid.b_momentum, grid.side, settings.b_rest,
                      min(settings.seal_compression, 3.0e38), settings.friction, float(dt)])
