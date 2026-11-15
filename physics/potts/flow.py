"""Flow: energy, momentum and heat move between neighbouring Planck boxes by the flow (donor cell, exact).

Contract: `flow(universe, settings, threshold)` moves energy across the box faces of the
boxes due this substep: a box sends only when its time level is at least `threshold`
(time_levels.py), for its own substep dt = step_time / 2**level times its cell's clock,
and every box with a due face neighbour takes in what is sent to it (the first
universe.scratch["flow_prefix"][threshold] boxes of "flow_order"). A box sends across
a face only to an occupied box, only when its flow velocity u = momentum / energy
points through that face, and sends floor(E * min(u * clock * dt, 1/6) * openness)
quanta of its E, openness from border_openness.py (1 inside a cell; through a joint
only at its seams, less the more it is squeezed, nothing past the seal). Both boxes
compute that same integer, one takes it and the other gives it, so total energy is
exact; a box can never send more than it holds. Momentum and heat go with the energy in
proportion; crossing an A-B border keeps 1 / (1 + friction * c) of the momentum, c the
joint's compression (the only friction), and the kinetic energy it loses, p**2 / 2e of
the crossing energy times (1 - kept**2), is booked as heat in the receiving box
(bible: friction turns directed flow into jiggling, heat; energy is untouched). Results
are written to the *_next buffers for the listed boxes and then copied back. Returns None.
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
def _sent(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
          momentum: wp.array3d(dtype=wp.vec3), kind: wp.array(dtype=wp.int32),
          compression: wp.array(dtype=wp.float32), seam: wp.array3d(dtype=wp.int32), clock: wp.array(dtype=wp.float32),
          box_level: wp.array3d(dtype=wp.int32), threshold: int, step_time: float, seal: float,
          here: wp.vec3i, face: int, side: int) -> wp.int64:
    """Quanta box `here` sends through `face` this substep; the same number both sides compute."""
    sender = box_owner(owner, here[0], here[1], here[2], side)
    if sender < 0:
        return wp.int64(0)
    level = box_level[here[0], here[1], here[2]]
    if level < threshold:
        return wp.int64(0)
    there = here + direction(face)
    if box_owner(owner, there[0], there[1], there[2], side) < 0:
        return wp.int64(0)
    held = energy[here[0], here[1], here[2]]
    if held <= wp.int64(0):
        return wp.int64(0)
    speed = momentum[here[0], here[1], here[2]][face // 2] * float(1 - 2 * (face % 2)) / (float(held) * _QUANTUM)
    if speed <= 0.0:
        return wp.int64(0)
    dt = step_time / wp.pow(2.0, float(level))
    share = wp.min(speed * clock[sender] * dt, 1.0 / 6.0) * border_openness(owner, kind, compression, seam, here, there,
                                                                             seal, side)
    return wp.int64(wp.float64(held) * wp.float64(share))


@wp.func
def _kept(kind: wp.array(dtype=wp.int32), compression: wp.array(dtype=wp.float32), sender: int, receiver: int,
          friction: float) -> float:
    """The share of momentum that survives crossing from sender's cell to receiver's."""
    if sender == receiver:
        return 1.0
    joint = receiver
    if kind[sender] == 1:
        joint = sender
    return 1.0 / (1.0 + friction * wp.max(compression[joint], 0.0))


@wp.kernel
def _flow(order: wp.array(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
          momentum: wp.array3d(dtype=wp.vec3), heat: wp.array3d(dtype=wp.float32), kind: wp.array(dtype=wp.int32),
          compression: wp.array(dtype=wp.float32), seam: wp.array3d(dtype=wp.int32), clock: wp.array(dtype=wp.float32),
          box_level: wp.array3d(dtype=wp.int32), threshold: int, step_time: float, seal: float, friction: float,
          side: int, energy_next: wp.array3d(dtype=wp.int64), momentum_next: wp.array3d(dtype=wp.vec3),
          heat_next: wp.array3d(dtype=wp.float32)):
    index = order[wp.tid()]
    i = index // (side * side)
    j = (index // side) % side
    k = index % side
    held = energy[i, j, k]
    carried = momentum[i, j, k]
    warmth = heat[i, j, k]
    cell = owner[i, j, k]
    here = wp.vec3i(i, j, k)
    kept = held
    new_momentum = carried
    new_heat = warmth
    for face in range(6):
        out = _sent(owner, energy, momentum, kind, compression, seam, clock, box_level, threshold, step_time, seal,
                    here, face, side)
        if out > wp.int64(0):
            fraction = float(wp.float64(out) / wp.float64(held))
            kept -= out
            new_momentum = new_momentum - carried * fraction
            new_heat = new_heat - warmth * fraction
        there = here + direction(face)
        incoming = _sent(owner, energy, momentum, kind, compression, seam, clock, box_level, threshold, step_time, seal,
                         there, face ^ 1, side)
        if incoming > wp.int64(0):
            source = owner[there[0], there[1], there[2]]
            share = float(wp.float64(incoming) / wp.float64(energy[there[0], there[1], there[2]]))
            arriving = momentum[there[0], there[1], there[2]] * share
            survives = _kept(kind, compression, source, cell, friction)
            new_momentum = new_momentum + arriving * survives
            new_heat = new_heat + heat[there[0], there[1], there[2]] * share
            if survives < 1.0:
                new_heat = new_heat + 0.5 * wp.dot(arriving, arriving) / (float(incoming) * _QUANTUM) * (
                    1.0 - survives * survives)
            kept += incoming
    energy_next[i, j, k] = kept
    momentum_next[i, j, k] = new_momentum
    heat_next[i, j, k] = new_heat


@wp.kernel
def _copy_back(order: wp.array(dtype=wp.int32), side: int, energy_next: wp.array3d(dtype=wp.int64),
               momentum_next: wp.array3d(dtype=wp.vec3), heat_next: wp.array3d(dtype=wp.float32),
               energy: wp.array3d(dtype=wp.int64), momentum: wp.array3d(dtype=wp.vec3),
               heat: wp.array3d(dtype=wp.float32)):
    index = order[wp.tid()]
    i = index // (side * side)
    j = (index // side) % side
    k = index % side
    energy[i, j, k] = energy_next[i, j, k]
    momentum[i, j, k] = momentum_next[i, j, k]
    heat[i, j, k] = heat_next[i, j, k]


def flow(universe: Universe, settings: Settings, threshold: int) -> None:
    """Move energy, momentum and heat across the faces of the due boxes, exactly."""
    count = universe.scratch["flow_prefix"][threshold]
    if count == 0:
        return
    device = universe.owner.device
    order = universe.scratch["flow_order"]
    wp.launch(_flow, dim=count, device=device,
              inputs=[order, universe.owner, universe.energy, universe.momentum, universe.heat, universe.kind,
                      universe.cell_compression, universe.seam, universe.cell_clock, universe.box_level, int(threshold),
                      settings.step_time, settings.seal_compression, settings.friction, universe.side,
                      universe.energy_next, universe.momentum_next, universe.heat_next])
    wp.launch(_copy_back, dim=count, device=device,
              inputs=[order, universe.side, universe.energy_next, universe.momentum_next, universe.heat_next,
                      universe.energy, universe.momentum, universe.heat])
