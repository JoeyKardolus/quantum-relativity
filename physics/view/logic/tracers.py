"""Tracers: dots of fluid that the cell's own flow carries, each with the way it came.

A tracer starts in a box drawn by its quanta, so the dots lie as the fluid lies. Before
each of a step's six passes it moves with the fluid of its box and that box's partner:
the engine's own pair rotation says how much crosses between the two, and the tracer
keeps as much of their fluid behind it as it had. Fluid that stands still moves no dot.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

from physics.engine import step
from physics.engine.axiom_1_energy.definitions.pair_rotation import TURN, pair_rotation, wave_of
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of

COUNT = 900  # tracers, a drawing number
MARKS = 16  # places a tracer remembers: where it is now and the 15 marks behind it


@dataclass
class Tracers:
    tail: wp.array  # vec3 (MARKS, tracers): places in boxes; row `head` is where each is now
    head: int = 0


@wp.kernel
def _carry(quanta: wp.array2d(dtype=wp.int32), phase: wp.array2d(dtype=wp.float32),
           slot: wp.int32, across: wp.int32, axis: wp.int32, parity: wp.int32,
           read_row: wp.int32, write_row: wp.int32, tail: wp.array2d(dtype=wp.vec3)):
    tracer = wp.tid()
    place = tail[read_row, tracer]
    strides = wp.vec3i(across * across, across, 1)
    inside = wp.vec3i(wp.clamp(int(place[0]), 0, across - 1),
                      wp.clamp(int(place[1]), 0, across - 1),
                      wp.clamp(int(place[2]), 0, across - 1))
    first = inside[axis] - (inside[axis] + parity) % 2  # the low box of this pass's pair
    if first >= 0 and first + 1 < across:
        low = wp.dot(inside, strides) + (first - inside[axis]) * strides[axis]
        high = low + strides[axis]
        low_has, high_has = float(quanta[slot, low]), float(quanta[slot, high])
        low_wave, high_wave = pair_rotation(wave_of(quanta[slot, low], phase[slot, low]),
                                            wave_of(quanta[slot, high], phase[slot, high]),
                                            wp.float64(TURN))
        stays = low_has + high_has - float(wp.dot(high_wave, high_wave))  # the low box, after
        line = place[axis] - float(first)  # 0 to 2 along the two boxes; `before` is the fluid
        before = wp.min(line, 1.0) * low_has + wp.max(line - 1.0, 0.0) * high_has  # behind it
        if before < stays:
            line = before / stays
        else:
            line = 1.0 + (before - stays) / wp.max(low_has + high_has - stays, 1.0e-6)
        place[axis] = float(first) + wp.clamp(line, 0.0, 2.0)
    tail[write_row, tracer] = place


def lay_tracers(store, cell: int, count: int = COUNT, seed: int = 1) -> Tracers:
    """`count` tracers at rest, laid over the cell's boxes by their quanta."""
    box_quanta, _phase, slot = boxes_of(store, cell)
    quanta, chance = box_quanta[slot].numpy().astype(float), np.random.default_rng(seed)
    boxes = chance.choice(quanta.size, size=count, p=quanta / quanta.sum())
    corner = np.stack(np.unravel_index(boxes, (store.across,) * 3), axis=1)
    place = (corner + chance.random((count, 3))).astype(np.float32)
    return Tracers(wp.array(np.tile(place, (MARKS, 1, 1)), dtype=wp.vec3, device=box_quanta.device))


def step_carrying(store, tracers: Tracers, cell: int, mark: bool) -> None:
    """One step of the engine with the tracers carried along; `mark` lengthens their tails."""
    new = (tracers.head + 1) % MARKS if mark else tracers.head
    box_quanta, box_phase, slot = boxes_of(store, cell)  # a calm cell's: the rule's, this step

    def carry(axis: int, parity: int) -> None:
        wp.launch(_carry, dim=tracers.tail.shape[1], device=box_quanta.device,
                  inputs=[box_quanta, box_phase, slot, store.across, axis, parity,
                          tracers.head, new, tracers.tail])
        tracers.head = new

    step(store, carry)
