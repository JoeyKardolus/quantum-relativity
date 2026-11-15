"""Flow: energy and momentum move between neighbouring Planck boxes by the flow (donor cell, exact).

Contract: `flow(universe, settings, dt)` moves energy across every box face for each
sending box's own time clock * dt and returns None. A box sends across a face only to
an occupied box, only when its flow velocity u = momentum / energy points through
that face, and sends floor(E * min(u * clock * dt, 1/6) * openness) quanta of its E,
openness from border_openness.py (1 inside a cell; through a joint less the more it
is squeezed, nothing past the seal). Both boxes compute that same integer, one takes
it and the other gives it, so total energy is exact; a box can never send more than
it holds. Momentum goes with the energy in proportion; crossing an A-B border keeps
1 / (1 + friction * c) of it, c the joint's compression (the only friction: directed
flow turned into undirected jiggling, heat, which is not tracked; energy is untouched).
Energy and momentum are computed into universe.energy_next / momentum_next and the
pairs are then swapped. Nothing moves into or out of an empty box.
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
          compression: wp.array(dtype=wp.float32), clock: wp.array(dtype=wp.float32), dt: float, seal: float,
          here: wp.vec3i, face: int, side: int) -> wp.int64:
    """Quanta box `here` sends through `face`; the same number both sides compute."""
    if here[0] < 0 or here[1] < 0 or here[2] < 0 or here[0] >= side or here[1] >= side or here[2] >= side:
        return wp.int64(0)
    sender = owner[here[0], here[1], here[2]]
    if sender < 0:
        return wp.int64(0)
    there = here + direction(face)
    receiver = box_owner(owner, there[0], there[1], there[2], side)
    if receiver < 0:
        return wp.int64(0)
    held = energy[here[0], here[1], here[2]]
    if held <= wp.int64(0):
        return wp.int64(0)
    speed = momentum[here[0], here[1], here[2]][face // 2] * float(1 - 2 * (face % 2)) / (float(held) * _QUANTUM)
    if speed <= 0.0:
        return wp.int64(0)
    share = wp.min(speed * clock[sender] * dt, 1.0 / 6.0) * border_openness(kind, compression, sender, receiver, seal)
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
def _flow(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
          momentum: wp.array3d(dtype=wp.vec3), kind: wp.array(dtype=wp.int32),
          compression: wp.array(dtype=wp.float32), clock: wp.array(dtype=wp.float32), dt: float, seal: float,
          friction: float, side: int, energy_next: wp.array3d(dtype=wp.int64),
          momentum_next: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    held = energy[i, j, k]
    carried = momentum[i, j, k]
    cell = owner[i, j, k]
    if cell < 0:
        energy_next[i, j, k] = held
        momentum_next[i, j, k] = carried
        return
    here = wp.vec3i(i, j, k)
    kept = held
    new_momentum = carried
    for face in range(6):
        out = _sent(owner, energy, momentum, kind, compression, clock, dt, seal, here, face, side)
        if out > wp.int64(0):
            kept -= out
            new_momentum = new_momentum - carried * float(wp.float64(out) / wp.float64(held))
        there = here + direction(face)
        incoming = _sent(owner, energy, momentum, kind, compression, clock, dt, seal, there, face ^ 1, side)
        if incoming > wp.int64(0):
            source = owner[there[0], there[1], there[2]]
            share = float(wp.float64(incoming) / wp.float64(energy[there[0], there[1], there[2]]))
            new_momentum = new_momentum + momentum[there[0], there[1], there[2]] * (
                share * _kept(kind, compression, source, cell, friction))
            kept += incoming
    energy_next[i, j, k] = kept
    momentum_next[i, j, k] = new_momentum


def flow(universe: Universe, settings: Settings, dt: float) -> None:
    """Move energy and momentum across every box face by the flow, exactly, then swap buffers."""
    wp.launch(_flow, dim=(universe.side,) * 3, device=universe.owner.device,
              inputs=[universe.owner, universe.energy, universe.momentum, universe.kind, universe.cell_compression,
                      universe.cell_clock, float(dt), settings.seal_compression, settings.friction, universe.side,
                      universe.energy_next, universe.momentum_next])
    universe.energy, universe.energy_next = universe.energy_next, universe.energy
    universe.momentum, universe.momentum_next = universe.momentum_next, universe.momentum
