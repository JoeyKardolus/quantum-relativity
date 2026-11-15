"""Kick: the pull and the fluid's own pressure accelerate the energy in every box.

Contract: `kick(universe, settings, dt)` adds to every occupied box's momentum, for its
cell's own time clock * dt (bible: time is local):
  the pull, -e * grad(potential), on A boxes only (A feels the pull, B does not),
  central differences, one-sided at the grid's faces;
  the pressure, -grad(P) along each axis, central differences, P = sound**2 * rest_a *
  e / rest of the box's kind (rest_a, rest_b, the start's mean per box): each kind's
  energy pressed against its own rest, as the clocks count it (cell_sums.py), so at
  the start's mean a joint box and an energy-volume box press alike and energy that
  crosses into a joint, where it is packed 1 / JOINT_SHARE times tighter, is pressed
  on to the next energy volume: the joint is a passage, not a store (bible: A to B
  to A). An empty neighbour or the grid's outside counts as a wall at the box's own
  pressure.
Then a closed face stops the flow into it: where the momentum points through a face
into an empty box, the grid's outside, or across a sealed joint border (openness 0,
border_openness.py), that component is set to 0, as at a hard wall (the directed flow
stopped there becomes heat, not tracked; no energy moves). Without this a pressure or
pull pressing on a closed face piles up momentum that can never flow.
e is the box's energy (quanta * QUANTUM). Empty boxes are left at zero. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.potts.border_openness import border_openness
from physics.potts.box_owner import box_owner
from physics.potts.direction import direction
from physics.potts.settings import Settings
from physics.potts.universe import QUANTUM, Universe

_QUANTUM = wp.constant(QUANTUM)


@wp.func
def _pressure(energy: wp.array3d(dtype=wp.int64), kind: wp.array(dtype=wp.int32), cell: int, i: int, j: int, k: int,
              sound_squared: float, rest_a: float, rest_b: float) -> float:
    rest = rest_a
    if kind[cell] == 1:
        rest = rest_b
    return sound_squared * rest_a * float(energy[i, j, k]) * _QUANTUM / rest


@wp.kernel
def _kick(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
          potential: wp.array3d(dtype=wp.float32), kind: wp.array(dtype=wp.int32),
          cell_clock: wp.array(dtype=wp.float32), compression: wp.array(dtype=wp.float32), sound_squared: float,
          rest_a: float, rest_b: float, seal: float, dt: float, side: int, momentum: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    cell = owner[i, j, k]
    if cell < 0:
        return
    held = float(energy[i, j, k]) * _QUANTUM
    here = wp.vec3i(i, j, k)
    force = wp.vec3(0.0, 0.0, 0.0)
    for axis in range(3):
        ahead = here + direction(2 * axis)
        behind = here + direction(2 * axis + 1)
        ahead_in = ahead[axis] < side
        behind_in = behind[axis] >= 0
        if kind[cell] == 0:
            upper = potential[i, j, k]
            lower = potential[i, j, k]
            span = 0.0
            if ahead_in:
                upper = potential[ahead[0], ahead[1], ahead[2]]
                span += 1.0
            if behind_in:
                lower = potential[behind[0], behind[1], behind[2]]
                span += 1.0
            if span > 0.0:
                force[axis] = force[axis] - held * (upper - lower) / span
        pressure_here = _pressure(energy, kind, cell, i, j, k, sound_squared, rest_a, rest_b)
        pressure_ahead = pressure_here
        pressure_behind = pressure_here
        ahead_cell = box_owner(owner, ahead[0], ahead[1], ahead[2], side)
        behind_cell = box_owner(owner, behind[0], behind[1], behind[2], side)
        if ahead_cell >= 0:
            pressure_ahead = _pressure(energy, kind, ahead_cell, ahead[0], ahead[1], ahead[2], sound_squared, rest_a,
                                       rest_b)
        if behind_cell >= 0:
            pressure_behind = _pressure(energy, kind, behind_cell, behind[0], behind[1], behind[2], sound_squared,
                                        rest_a, rest_b)
        force[axis] = force[axis] - 0.5 * (pressure_ahead - pressure_behind)
    pushed = momentum[i, j, k] + force * (dt * cell_clock[cell])
    for axis in range(3):
        face = 2 * axis
        if pushed[axis] < 0.0:
            face = 2 * axis + 1
        there = here + direction(face)
        beyond = box_owner(owner, there[0], there[1], there[2], side)
        if beyond < 0:
            pushed[axis] = 0.0
        elif border_openness(kind, compression, cell, beyond, seal) <= 0.0:
            pushed[axis] = 0.0
    momentum[i, j, k] = pushed


def kick(universe: Universe, settings: Settings, dt: float) -> None:
    """Accelerate every box's energy by the pull (A only) and the pressure of each kind against its rest."""
    wp.launch(_kick, dim=(universe.side,) * 3, device=universe.owner.device,
              inputs=[universe.owner, universe.energy, universe.potential, universe.kind, universe.cell_clock,
                      universe.cell_compression, settings.sound * settings.sound, universe.rest_a, universe.rest_b,
                      settings.seal_compression, float(dt), universe.side, universe.momentum])
