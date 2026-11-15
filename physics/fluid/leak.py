"""Leak: A-energy moves from box to box through the joints between them, split by pull (drift).

Contract: `leak(grid, settings, dt)` moves A-energy, and the A momentum it carries,
across the faces between neighbouring boxes, in place (bible foundation 4 and rule 6;
the earlier ABAB engine's leak.py, git history, on boxes). A face is a joint when
both its boxes hold B above settings.cfl_floor * grid.start_a_mean; its openness is
joint_openness.py of the larger of the two boxes' B compression (b / b_rest), so a
box at or past the seal point leaks nothing and takes nothing. Per box i:
  outflow_i = mean openness of its joint faces (0 with none)
  given_i   = settings.leak * a_i * outflow_i * clock_i * dt   (its own time, clock_rate.py)
  to j      = given_i * a_j * open_ij / sum over i's joint faces of a_j' * open_ij'
a the A-energy per box: a neighbour's pull is its A-energy (a box is a fixed set of
volumes, local_length.py, so more A is a smaller, more packed volume that pulls
harder; bible rule 7). Pulls of 4, 2, 1, 1 take 4/8, 2/8, 1/8, 1/8. Between even
neighbours the flows cancel; a box between a packed and a thin neighbour sends most
to the packed one, so lumps gain and the thin side loses energy and grows longer
(local_length.py): space grows as lumps form, with no separate drift setting. Each
face moves one integer of quanta, taken from one box and given to the other (exact),
and the leaked energy keeps its giver's A flow velocity (momentum moves with it, none
is made). The B-energy is never changed: a joint is the passage, not a store. A box
gives at most leak * clock * dt of its A, within its content at the CFL substep
(fastest_wave.py counts the leak). Buffers are per (side, device) and reused.
"""
from __future__ import annotations

import warp as wp

from physics.fluid.flow_velocity import flow_velocity
from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.joint_openness import joint_openness
from physics.fluid.settings import Settings

_QUANTUM = wp.constant(QUANTUM)
_BUFFERS: dict[tuple[int, str], list] = {}


@wp.func
def _amount(quanta: wp.int64) -> float:
    return float(wp.max(quanta, wp.int64(0))) * _QUANTUM


@wp.func
def _face_open(b_energy: wp.array3d(dtype=wp.int64), here: wp.vec3i, there: wp.vec3i, b_rest: float, seal: float,
               floor: float) -> float:
    """The openness of the joint between two face neighbours, 0 when either holds no B."""
    b_here = _amount(b_energy[here[0], here[1], here[2]])
    b_there = _amount(b_energy[there[0], there[1], there[2]])
    if b_here <= floor or b_there <= floor:
        return 0.0
    return joint_openness(wp.max(b_here, b_there) / b_rest, seal)


@wp.kernel
def _given(a_energy: wp.array3d(dtype=wp.int64), b_energy: wp.array3d(dtype=wp.int64),
           clock: wp.array3d(dtype=float), side: int, b_rest: float, seal: float, floor: float, rate: float,
           dt: float, per_pull: wp.array3d(dtype=float)):
    """Each box's given A-energy over the sum of its takers' pull times openness."""
    i, j, k = wp.tid()
    here = wp.vec3i(i, j, k)
    opens = float(0.0)
    joints = float(0.0)
    around = float(0.0)
    for axis in range(3):
        for towards in range(2):
            there = wp.vec3i(i, j, k)
            there[axis] = there[axis] + 2 * towards - 1
            if there[axis] >= 0 and there[axis] < side:
                if _amount(b_energy[i, j, k]) > floor and _amount(b_energy[there[0], there[1], there[2]]) > floor:
                    openness = _face_open(b_energy, here, there, b_rest, seal, floor)
                    joints += 1.0
                    opens += openness
                    around += _amount(a_energy[there[0], there[1], there[2]]) * openness
    per_pull[i, j, k] = 0.0
    if joints > 0.0 and around > 0.0:
        given = rate * _amount(a_energy[i, j, k]) * (opens / joints) * clock[i, j, k] * dt
        per_pull[i, j, k] = given / around


@wp.kernel
def _face(a_energy: wp.array3d(dtype=wp.int64), b_energy: wp.array3d(dtype=wp.int64),
          a_momentum: wp.array3d(dtype=wp.vec3), per_pull: wp.array3d(dtype=float), axis: int, side: int,
          b_rest: float, seal: float, floor: float,
          flux: wp.array3d(dtype=wp.int64), carry: wp.array3d(dtype=wp.vec3)):
    """Net A leak in +axis over the face at index (i, j, k): between box index - axis unit and box index."""
    i, j, k = wp.tid()
    left = wp.vec3i(i, j, k)
    left[axis] = left[axis] - 1
    right = wp.vec3i(i, j, k)
    flux[i, j, k] = wp.int64(0)
    carry[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    if left[axis] < 0 or left[axis] + 1 >= side:
        return
    openness = _face_open(b_energy, left, right, b_rest, seal, floor)
    if openness <= 0.0:
        return
    forward = per_pull[left[0], left[1], left[2]] * _amount(a_energy[i, j, k]) * openness
    back = per_pull[i, j, k] * _amount(a_energy[left[0], left[1], left[2]]) * openness
    flux[i, j, k] = wp.int64(wp.round(wp.float64(forward - back) / wp.float64(_QUANTUM)))
    giver = flow_velocity(a_momentum[left[0], left[1], left[2]], a_energy[left[0], left[1], left[2]])
    carry[i, j, k] = (forward * giver
                      - back * flow_velocity(a_momentum[i, j, k], a_energy[i, j, k]))


@wp.kernel
def _apply(a_energy: wp.array3d(dtype=wp.int64), a_momentum: wp.array3d(dtype=wp.vec3),
           flux_x: wp.array3d(dtype=wp.int64), flux_y: wp.array3d(dtype=wp.int64),
           flux_z: wp.array3d(dtype=wp.int64), carry_x: wp.array3d(dtype=wp.vec3),
           carry_y: wp.array3d(dtype=wp.vec3), carry_z: wp.array3d(dtype=wp.vec3)):
    """Each box takes what leaks across its low faces and gives what leaks across its high faces."""
    i, j, k = wp.tid()
    a_energy[i, j, k] = (a_energy[i, j, k] + flux_x[i, j, k] - flux_x[i + 1, j, k] + flux_y[i, j, k]
                         - flux_y[i, j + 1, k] + flux_z[i, j, k] - flux_z[i, j, k + 1])
    a_momentum[i, j, k] = (a_momentum[i, j, k] + carry_x[i, j, k] - carry_x[i + 1, j, k] + carry_y[i, j, k]
                           - carry_y[i, j + 1, k] + carry_z[i, j, k] - carry_z[i, j, k + 1])


def leak(grid: Grid, settings: Settings, dt: float) -> None:
    """Leak A-energy through the open joints to the neighbours, by their pull, for dt."""
    device = str(grid.a_energy.device)
    side = grid.side
    key = (side, device)
    if key not in _BUFFERS:
        shapes = [(side + 1, side, side), (side, side + 1, side), (side, side, side + 1)]
        _BUFFERS[key] = [wp.zeros((side,) * 3, dtype=float, device=device),
                         [wp.zeros(shape, dtype=wp.int64, device=device) for shape in shapes],
                         [wp.zeros(shape, dtype=wp.vec3, device=device) for shape in shapes]]
    per_pull, flux, carry = _BUFFERS[key]
    seal = min(settings.seal_compression, 3.0e38)
    floor = settings.cfl_floor * grid.start_a_mean
    wp.launch(_given, dim=(side,) * 3, device=device,
              inputs=[grid.a_energy, grid.b_energy, grid.clock, side, settings.b_rest, seal, floor, settings.leak,
                      float(dt), per_pull])
    for axis in range(3):
        wp.launch(_face, dim=flux[axis].shape, device=device,
                  inputs=[grid.a_energy, grid.b_energy, grid.a_momentum, per_pull, axis, side, settings.b_rest, seal,
                          floor, flux[axis], carry[axis]])
    wp.launch(_apply, dim=(side,) * 3, device=device, inputs=[grid.a_energy, grid.a_momentum, *flux, *carry])
