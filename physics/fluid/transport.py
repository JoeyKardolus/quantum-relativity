"""Transport: energy and momentum flow across the faces between boxes for one substep.

Contract: `transport(grid, settings, dt)` moves A-energy, B-energy, both momenta and
heat across every face between neighbouring boxes, in place. Each box sends in its
own time: its flux is multiplied by its clock rate (clock_rate.py), so a slow packed
box sends slowly. A's flux on a face is further multiplied by the receiving box's A
passage (a_passage.py, a_rest = grid.start_a_mean, b_rest = settings.b_rest, floor
settings.cfl_floor * grid.start_a_mean): A flows only through joints and never packs
without B between. The momentum the held-back A would have carried is given to the
receiving box instead, split between its A and B by their energy (the push of A
against the joints that stop it): momentum is kept, and held A cannot speed up for
ever against a face it cannot pass. Heat moves with the energy it sits in: a box sends the share
of its heat that it sends of its A- plus B-energy. Numerics: first-order
finite volume, unsplit (all three axes from the same state), donor cell: each box
sends, through each face, e * max(u_n, 0) * dt of each kind's energy e by its own
flow velocity u (flow_velocity.py; u_n towards that face), and the momentum that
energy carries, sent energy * u. So the energy moved is exactly the momentum held
(sum e u dt): a blob's centre moves at its total momentum over its energy, whatever
its inside does. (Rusanov fluxes were tried first: their wave-speed diffusion is not
Galilean, and a collapsing blob moving at 0.5 slowed to 0.34 in 20 time units while
its momentum stayed, measured.) Pressure is not here (push_kick.py). Energy fluxes
are rounded to whole quanta (grid.py) and the same integer is taken from one box and
given to the other: the sum of all energy is exact. The seal needs nothing here:
rub.py, run just before, gives A the velocity of its B on both sides of every face
that touches a sealed box, so A crosses such a face only inside the B that crosses
it, at that box's A-to-B share (zero A flux in B's frame). A sealed lump so keeps
its A while it moves or sits and loses A only as its own B flows out. (A flux zero
in the grid's frame was tried first: it pins a moving sealed lump to the grid; a
per-face share rule next: the A it moved at B's velocity kept A's own momentum, and
a moving blob that sealed ran 3.5% fast, measured.) Walls: nothing crosses the grid's outer faces. The substep
keeps clock * dt * fastest_wave <= courant (step.py) for every box above the CFL floor
(fastest_wave.py), which keeps such a box's outflow within its content up to a few
quanta of rounding (read as 0). A box whose outflow over dt would pass its content
(dt times the sum over axes of its sending speed, sending_speed.py, above 1: only near-empty vapour below
the floor) sends each kind at the share 1 / (dt * that sum) of its donor-cell flux on
every face, momentum with it, so it sends at most what it holds and keeps its flow
velocity; every box that sets the substep sends at share 1, unchanged. Buffers are
per (side, device) and reused.
"""
from __future__ import annotations

import warp as wp

from physics.fluid.a_passage import a_passage
from physics.fluid.flow_velocity import flow_velocity
from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.sending_speed import sending_speed
from physics.fluid.settings import Settings

_QUANTUM = wp.constant(QUANTUM)
_BUFFERS: dict[tuple[int, str], list] = {}


@wp.func
def _energy(quanta: wp.int64) -> float:
    return float(wp.max(quanta, wp.int64(0))) * _QUANTUM


@wp.func
def _quanta(amount: float) -> wp.int64:
    return wp.int64(wp.round(wp.float64(amount) / wp.float64(_QUANTUM)))


@wp.func
def _sent(energy: float, velocity: wp.vec3, axis: int, towards: float) -> float:
    """Energy per time a box sends through its face on side `towards` (+1 or -1) of axis."""
    return energy * wp.max(towards * velocity[axis], 0.0)


@wp.func
def _share(velocity: wp.vec3, box: wp.vec3i, side: int, dt: float) -> float:
    """The share of its donor-cell flux a box sends: 1, or less when that would pass its content."""
    outflow = dt * (sending_speed(velocity, 0, box[0], side) + sending_speed(velocity, 1, box[1], side)
                    + sending_speed(velocity, 2, box[2], side))
    if outflow <= 1.0:
        return 1.0
    return 1.0 / outflow


@wp.func
def _heat_sent(heat: float, a_sent: float, b_sent: float, a_held: float, b_held: float) -> float:
    """The heat that goes with the energy a box sends: the same share of its heat."""
    held = a_held + b_held
    if held <= 0.0:
        return 0.0
    return heat * (a_sent + b_sent) / held


@wp.kernel
def _face(a_energy: wp.array3d(dtype=wp.int64), b_energy: wp.array3d(dtype=wp.int64),
          a_momentum: wp.array3d(dtype=wp.vec3), b_momentum: wp.array3d(dtype=wp.vec3),
          clock: wp.array3d(dtype=float), heat: wp.array3d(dtype=float),
          axis: int, side: int, dt: float, a_rest: float, b_rest: float, floor: float,
          a_flux: wp.array3d(dtype=wp.int64), b_flux: wp.array3d(dtype=wp.int64),
          a_carry: wp.array3d(dtype=wp.vec3), b_carry: wp.array3d(dtype=wp.vec3),
          heat_flux: wp.array3d(dtype=float), held_forward: wp.array3d(dtype=wp.vec3),
          held_back: wp.array3d(dtype=wp.vec3)):
    """Net flow in +axis over the face at index (i, j, k): between box index - axis unit and box index."""
    i, j, k = wp.tid()
    left = wp.vec3i(i, j, k)
    left[axis] = left[axis] - 1
    a_flux[i, j, k] = wp.int64(0)
    b_flux[i, j, k] = wp.int64(0)
    a_carry[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    b_carry[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    heat_flux[i, j, k] = 0.0
    held_forward[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    held_back[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    if left[axis] < 0 or left[axis] + 1 >= side:
        return
    a_left = _energy(a_energy[left[0], left[1], left[2]])
    a_right = _energy(a_energy[i, j, k])
    b_left = _energy(b_energy[left[0], left[1], left[2]])
    b_right = _energy(b_energy[i, j, k])
    clock_left = clock[left[0], left[1], left[2]]
    clock_right = clock[i, j, k]
    ua_left = flow_velocity(a_momentum[left[0], left[1], left[2]], a_energy[left[0], left[1], left[2]])
    ua_right = flow_velocity(a_momentum[i, j, k], a_energy[i, j, k])
    ub_left = flow_velocity(b_momentum[left[0], left[1], left[2]], b_energy[left[0], left[1], left[2]])
    ub_right = flow_velocity(b_momentum[i, j, k], b_energy[i, j, k])
    b_forward = (_sent(b_left, ub_left, axis, 1.0) * _share(ub_left, left, side, clock_left * dt)
                 * clock_left)
    b_back = (_sent(b_right, ub_right, axis, -1.0) * _share(ub_right, wp.vec3i(i, j, k), side, clock_right * dt)
              * clock_right)
    b_flux[i, j, k] = _quanta((b_forward - b_back) * dt)
    b_carry[i, j, k] = (b_forward * ub_left - b_back * ub_right) * dt
    into_right = a_passage(a_right, b_right, a_rest, b_rest, floor)
    into_left = a_passage(a_left, b_left, a_rest, b_rest, floor)
    a_forward_all = (_sent(a_left, ua_left, axis, 1.0) * _share(ua_left, left, side, clock_left * dt)
                     * clock_left)
    a_back_all = (_sent(a_right, ua_right, axis, -1.0) * _share(ua_right, wp.vec3i(i, j, k), side,
                                                                clock_right * dt) * clock_right)
    a_forward = a_forward_all * into_right
    a_back = a_back_all * into_left
    held_forward[i, j, k] = (a_forward_all - a_forward) * ua_left * dt
    held_back[i, j, k] = (a_back_all - a_back) * ua_right * dt
    a_flux[i, j, k] = _quanta((a_forward - a_back) * dt)
    a_carry[i, j, k] = (a_forward * ua_left - a_back * ua_right) * dt
    heat_flux[i, j, k] = (_heat_sent(heat[left[0], left[1], left[2]], a_forward, b_forward, a_left, b_left)
                          - _heat_sent(heat[i, j, k], a_back, b_back, a_right, b_right)) * dt


@wp.kernel
def _apply(a_energy: wp.array3d(dtype=wp.int64), b_energy: wp.array3d(dtype=wp.int64),
           a_momentum: wp.array3d(dtype=wp.vec3), b_momentum: wp.array3d(dtype=wp.vec3),
           a_flux_x: wp.array3d(dtype=wp.int64), a_flux_y: wp.array3d(dtype=wp.int64),
           a_flux_z: wp.array3d(dtype=wp.int64), b_flux_x: wp.array3d(dtype=wp.int64),
           b_flux_y: wp.array3d(dtype=wp.int64), b_flux_z: wp.array3d(dtype=wp.int64),
           a_carry_x: wp.array3d(dtype=wp.vec3), a_carry_y: wp.array3d(dtype=wp.vec3),
           a_carry_z: wp.array3d(dtype=wp.vec3), b_carry_x: wp.array3d(dtype=wp.vec3),
           b_carry_y: wp.array3d(dtype=wp.vec3), b_carry_z: wp.array3d(dtype=wp.vec3),
           heat: wp.array3d(dtype=float), heat_x: wp.array3d(dtype=float), heat_y: wp.array3d(dtype=float),
           heat_z: wp.array3d(dtype=float), held_forward_x: wp.array3d(dtype=wp.vec3),
           held_forward_y: wp.array3d(dtype=wp.vec3), held_forward_z: wp.array3d(dtype=wp.vec3),
           held_back_x: wp.array3d(dtype=wp.vec3), held_back_y: wp.array3d(dtype=wp.vec3),
           held_back_z: wp.array3d(dtype=wp.vec3)):
    """Each box takes what crosses its low faces and gives what crosses its high faces."""
    i, j, k = wp.tid()
    held_in = (held_forward_x[i, j, k] + held_forward_y[i, j, k] + held_forward_z[i, j, k]
               + held_back_x[i + 1, j, k] + held_back_y[i, j + 1, k] + held_back_z[i, j, k + 1])
    held_out = (held_forward_x[i + 1, j, k] + held_forward_y[i, j + 1, k] + held_forward_z[i, j, k + 1]
                + held_back_x[i, j, k] + held_back_y[i, j, k] + held_back_z[i, j, k])
    pulling = _energy(a_energy[i, j, k])
    pushing = _energy(b_energy[i, j, k])
    a_part = 1.0
    if pulling + pushing > 0.0:
        a_part = pulling / (pulling + pushing)
    heat[i, j, k] = (heat[i, j, k] + heat_x[i, j, k] - heat_x[i + 1, j, k] + heat_y[i, j, k] - heat_y[i, j + 1, k]
                     + heat_z[i, j, k] - heat_z[i, j, k + 1])
    a_energy[i, j, k] = (a_energy[i, j, k] + a_flux_x[i, j, k] - a_flux_x[i + 1, j, k]
                         + a_flux_y[i, j, k] - a_flux_y[i, j + 1, k]
                         + a_flux_z[i, j, k] - a_flux_z[i, j, k + 1])
    b_energy[i, j, k] = (b_energy[i, j, k] + b_flux_x[i, j, k] - b_flux_x[i + 1, j, k]
                         + b_flux_y[i, j, k] - b_flux_y[i, j + 1, k]
                         + b_flux_z[i, j, k] - b_flux_z[i, j, k + 1])
    a_momentum[i, j, k] = (a_momentum[i, j, k] - held_out + a_part * held_in
                           + a_carry_x[i, j, k] - a_carry_x[i + 1, j, k]
                           + a_carry_y[i, j, k] - a_carry_y[i, j + 1, k]
                           + a_carry_z[i, j, k] - a_carry_z[i, j, k + 1])
    b_momentum[i, j, k] = (b_momentum[i, j, k] + (1.0 - a_part) * held_in
                           + b_carry_x[i, j, k] - b_carry_x[i + 1, j, k]
                           + b_carry_y[i, j, k] - b_carry_y[i, j + 1, k]
                           + b_carry_z[i, j, k] - b_carry_z[i, j, k + 1])


def transport(grid: Grid, settings: Settings, dt: float) -> None:
    """Move energy, momentum and heat across every face for one substep of dt, each box in its own time."""
    device = str(grid.a_energy.device)
    side = grid.side
    key = (side, device)
    if key not in _BUFFERS:
        shapes = [(side + 1, side, side), (side, side + 1, side), (side, side, side + 1)]
        _BUFFERS[key] = [[wp.zeros(shape, dtype=dtype, device=device) for shape in shapes]
                         for dtype in (wp.int64, wp.int64, wp.vec3, wp.vec3, float, wp.vec3, wp.vec3)]
    a_flux, b_flux, a_carry, b_carry, heat_flux, held_forward, held_back = _BUFFERS[key]
    for axis in range(3):
        wp.launch(_face, dim=a_flux[axis].shape, device=device,
                  inputs=[grid.a_energy, grid.b_energy, grid.a_momentum, grid.b_momentum, grid.clock, grid.heat,
                          axis, side, float(dt), grid.start_a_mean, settings.b_rest,
                          settings.cfl_floor * grid.start_a_mean, a_flux[axis], b_flux[axis], a_carry[axis],
                          b_carry[axis], heat_flux[axis], held_forward[axis], held_back[axis]])
    wp.launch(_apply, dim=(side, side, side), device=device,
              inputs=[grid.a_energy, grid.b_energy, grid.a_momentum, grid.b_momentum,
                      *a_flux, *b_flux, *a_carry, *b_carry, grid.heat, *heat_flux, *held_forward, *held_back])
