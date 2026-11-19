"""Tracers: dots of fluid that the cells' own flow carries, each with the way it came.

A tracer starts in a box drawn by its quanta (`lay_tracers.py`); its place is (layer,
around the core, along the ring) in boxes. Before each of a step's passes it moves with
the fluid of its box and that box's partner in that pass, the engine's own pairs: rows
closed on themselves, and from one ring to the next only in the jet. The engine's pair
rotation says how much crosses between the two, and the tracer keeps as much of their
fluid behind it as it had. Fluid that stands still moves no dot.
With several cells a place's second number says which slot's cell: `across` a slot.
"""
from dataclasses import dataclass

import warp as wp

from physics.engine import step
from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, wave_of
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.view.definitions.carried import carried

COUNT = 8000  # tracers a cell, a drawing number
MARKS, MARK_EVERY = 32, 4  # places a tracer remembers (now and 31 behind); steps a mark


@dataclass
class Tracers:
    tail: wp.array  # vec3 (MARKS, tracers): places in boxes; row `head` is where each is now
    head: int = 0


@wp.kernel
def _carry(quanta: wp.array2d(dtype=wp.int32), phase: wp.array2d(dtype=wp.float32),
           slot: wp.int32, cells: wp.int32, across: wp.int32, axis: wp.int32, parity: wp.int32,
           turn: wp.float64,
           read_row: wp.int32, write_row: wp.int32, tail: wp.array2d(dtype=wp.vec3)):
    tracer = wp.tid()
    place = tail[read_row, tracer]
    cell = wp.clamp(int(place[1]) / across, 0, cells - 1)
    place[1] -= float(cell * across)
    strides = wp.vec3i(across * across, across, 1)
    inside = wp.vec3i(wp.clamp(int(place[0]), 0, across - 1),
                      wp.clamp(int(place[1]), 0, across - 1),
                      wp.clamp(int(place[2]), 0, across - 1))
    first = (inside[axis] - (inside[axis] + parity) % 2 + across) % across  # the pair's low box
    last = (first + 1) % across
    jet = inside[1] == 0 or inside[1] == across - 1  # where one ring's fluid meets the next's
    if axis != 0 or (first + 1) % (across / 3) != 0 or jet:
        low = wp.dot(inside, strides) + (first - inside[axis]) * strides[axis]
        high = low + (last - first) * strides[axis]
        own = slot + cell  # the cells' slots follow each other
        low_has, high_has = wp.abs(quanta[own, low]), wp.abs(quanta[own, high])  # minus too
        low_wave, high_wave = pair_rotation(wave_of(low_has, phase[own, low]),
                                            wave_of(high_has, phase[own, high]), turn)
        stays = float(low_has + high_has) - float(wp.dot(high_wave, high_wave))  # low, after
        line = place[axis] - float(inside[axis]) + float(inside[axis] == last)  # along the pair
        line = carried(line, float(low_has), float(high_has), stays)
        place[axis] = line + float(first)
        if line >= 1.0:
            place[axis] = line - 1.0 + float(last)
    place[1] += float(cell * across)
    tail[write_row, tracer] = place


def step_carrying(store, tracers: Tracers, cell: int, mark: bool, boxes: tuple = None,
                  cells: int = 1) -> None:
    """One step of the engine with the tracers carried along; `mark` lengthens their tails."""
    # `boxes` saves asking again: a calm cell's stay good, its flow is the same every step
    new = (tracers.head + 1) % MARKS if mark else tracers.head
    line = (*(boxes or boxes_of(store, cell)), cells, store.across)
    def carry(axis: int, parity: int, turn: float) -> None:
        wp.launch(_carry, dim=tracers.tail.shape[1], device=tracers.tail.device,
                  inputs=[*line, axis, parity, wp.float64(turn), tracers.head, new, tracers.tail])
        tracers.head = new
    step(store, carry)
