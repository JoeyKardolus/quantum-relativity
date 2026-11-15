"""Fastest flow: the largest (|u_x| + |u_y| + |u_z| + sound) * clock over the boxes, for the substep.

Contract: `fastest_flow(universe, settings)` returns, as a Python float, the maximum
over occupied boxes holding energy of the sum of the flow speed's three components
plus the sound speed of its kind (sound * sqrt(rest_a / rest of its kind): energy in
a joint, pressed against its smaller rest, rings faster; kick.py), times the box's
cell clock (each region in its own time). Boxes holding under VAPOUR (a tenth) of
their cell's mean energy per box are left out: such vapour, a box just claimed and filling, gets a large kick for its little energy, and
flow.py lets any box send at most a sixth of what it holds per face, so it cannot go
negative; the substep is set by content, not vapour (as in the earlier fluid engine).
One small copy to the host. Raises ValueError when a speed is not finite.
"""
from __future__ import annotations

import math

import warp as wp

from physics.potts.settings import Settings
from physics.potts.universe import QUANTUM, Universe

_QUANTUM = wp.constant(QUANTUM)
VAPOUR = 0.1


@wp.kernel
def _fastest(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
             momentum: wp.array3d(dtype=wp.vec3), clock: wp.array(dtype=wp.float32),
             cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32),
             kind: wp.array(dtype=wp.int32), vapour: float, sound_a: float, sound_b: float,
             out: wp.array(dtype=wp.float32)):
    i, j, k = wp.tid()
    cell = owner[i, j, k]
    if cell < 0:
        return
    held = float(energy[i, j, k]) * _QUANTUM
    mean = float(cell_energy[cell]) * _QUANTUM / float(wp.max(cell_boxes[cell], 1))
    if held <= vapour * mean or held <= 0.0:
        return
    flow = momentum[i, j, k] / held
    sound = sound_a
    if kind[cell] == 1:
        sound = sound_b
    speed = (wp.abs(flow[0]) + wp.abs(flow[1]) + wp.abs(flow[2]) + sound) * clock[cell]
    if not wp.isfinite(speed):
        speed = 1.0e30
    wp.atomic_max(out, 0, speed)


def fastest_flow(universe: Universe, settings: Settings) -> float:
    """The fastest flow plus sound speed, each box in its cell's time."""
    out = wp.zeros(1, dtype=wp.float32, device=universe.owner.device)
    wp.launch(_fastest, dim=(universe.side,) * 3, device=universe.owner.device,
              inputs=[universe.owner, universe.energy, universe.momentum, universe.cell_clock, universe.cell_energy,
                      universe.cell_boxes, universe.kind, VAPOUR, settings.sound,
                      settings.sound * math.sqrt(universe.rest_a / universe.rest_b), out])
    speed = float(out.numpy()[0])
    if not math.isfinite(speed) or speed >= 1.0e30:
        raise ValueError(f"step {universe.steps + 1}: a flow speed is not finite")
    return speed
