"""Kick: the pull and the fluid's own pressure accelerate the energy in the boxes due this substep.

Contract: `kick(universe, settings, threshold)` acts on the boxes whose time level is at
least `threshold` (the first universe.scratch["kick_prefix"][threshold] boxes of
"kick_order", time_levels.py), each for its own substep dt = step_time / 2**level times
its cell's clock (bible: time is local), adding to its momentum:
  the pull, -e * grad(potential), on A boxes only (A feels the pull, B does not),
  central differences, one-sided at the grid's faces;
  the pressure, P = sound**2 * e for both kinds (a joint is a full cell holding the same
  fluid; bible rule 8), as the force e * sound**2 * grad(ln e) along each axis, central
  differences of ln e (the same force as grad P where the fluid is smooth; written in ln e
  a box next to a nearly empty one is not flung at 1e9 boxes per time, measured with
  grad P: boxes of a few quanta beside full ones). An empty neighbour or the grid's
  outside counts as a wall at the box's own energy.
Closed faces: along each axis, where the momentum points through a face of openness o
(border_openness.py: 0 into an empty box, the outside, a sealed joint border or a
joint's middle; graded through a squeezed seam; 1 inside the cell), the share (1 - o) *
min(1, |u| * dt * clock) of that component (the part of the flow reaching the face this
substep) is stopped, as at a hard wall, and its kinetic energy (p**2 / 2e lost) is
booked as the box's heat (bible: friction makes heat; no energy quantum moves). The
force's component into a face is then scaled by its openness: the closed part of a
face pushes back as a wall does, so pressing on a wall makes no flow and no heat.
Stopping only at fully closed faces let momentum pile up against a nearly sealed seam
to speeds of hundreds of boxes per time, which set every substep (measured 2026-11-15,
10^3 cells: the fastest box sat at 0.1 of its cell's mean next to a joint at
compression 4 to 5).
e is the box's energy (quanta * QUANTUM). Returns None.
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
def _log_energy(energy: wp.array3d(dtype=wp.int64), place: wp.vec3i) -> float:
    return wp.log(float(wp.max(energy[place[0], place[1], place[2]], wp.int64(1))))


@wp.kernel
def _kick(order: wp.array(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
          potential: wp.array3d(dtype=wp.float32), kind: wp.array(dtype=wp.int32),
          cell_clock: wp.array(dtype=wp.float32), compression: wp.array(dtype=wp.float32),
          seam: wp.array3d(dtype=wp.int32), box_level: wp.array3d(dtype=wp.int32), sound_squared: float, seal: float,
          step_time: float, side: int,
          momentum: wp.array3d(dtype=wp.vec3), heat: wp.array3d(dtype=wp.float32)):
    index = order[wp.tid()]
    i = index // (side * side)
    j = (index // side) % side
    k = index % side
    cell = owner[i, j, k]
    held = float(energy[i, j, k]) * _QUANTUM
    proper = step_time / wp.pow(2.0, float(box_level[i, j, k])) * cell_clock[cell]
    here = wp.vec3i(i, j, k)
    force = wp.vec3(0.0, 0.0, 0.0)
    log_here = _log_energy(energy, here)
    for axis in range(3):
        ahead = here + direction(2 * axis)
        behind = here + direction(2 * axis + 1)
        if kind[cell] == 0:
            upper = potential[i, j, k]
            lower = potential[i, j, k]
            span = 0.0
            if ahead[axis] < side:
                upper = potential[ahead[0], ahead[1], ahead[2]]
                span += 1.0
            if behind[axis] >= 0:
                lower = potential[behind[0], behind[1], behind[2]]
                span += 1.0
            if span > 0.0:
                force[axis] = force[axis] - held * (upper - lower) / span
        log_ahead = log_here
        log_behind = log_here
        if box_owner(owner, ahead[0], ahead[1], ahead[2], side) >= 0:
            log_ahead = _log_energy(energy, ahead)
        if box_owner(owner, behind[0], behind[1], behind[2], side) >= 0:
            log_behind = _log_energy(energy, behind)
        force[axis] = force[axis] - 0.5 * sound_squared * held * (log_ahead - log_behind)
    pushed = momentum[i, j, k]
    lost = float(0.0)
    if held > 0.0:
        for axis in range(3):
            face = 2 * axis
            if pushed[axis] < 0.0:
                face = 2 * axis + 1
            openness = border_openness(owner, kind, compression, seam, here, here + direction(face), seal, side)
            if openness < 1.0:
                reach = wp.min(1.0, wp.abs(pushed[axis]) / held * proper)
                kept = pushed[axis] * (1.0 - (1.0 - openness) * reach)
                lost += 0.5 * (pushed[axis] * pushed[axis] - kept * kept) / held
                pushed[axis] = kept
            towards = 2 * axis
            if force[axis] < 0.0:
                towards = 2 * axis + 1
            force[axis] = force[axis] * border_openness(owner, kind, compression, seam, here, here + direction(towards),
                                                        seal, side)
        pushed = pushed + force * proper
    else:
        pushed = wp.vec3(0.0, 0.0, 0.0)
    momentum[i, j, k] = pushed
    heat[i, j, k] = heat[i, j, k] + lost


def kick(universe: Universe, settings: Settings, threshold: int) -> None:
    """Accelerate the due boxes' energy by the pull (A only) and the pressure; stop flow at closed faces."""
    count = universe.scratch["kick_prefix"][threshold]
    if count == 0:
        return
    wp.launch(_kick, dim=count, device=universe.owner.device,
              inputs=[universe.scratch["kick_order"], universe.owner, universe.energy, universe.potential,
                      universe.kind, universe.cell_clock, universe.cell_compression, universe.seam,
                      universe.box_level,
                      settings.sound * settings.sound, settings.seal_compression, settings.step_time, universe.side,
                      universe.momentum, universe.heat])
