"""Tracers across: a tracer in a volume's box that faces a joint is carried across with the fluid.

The engine's own passage says how much crosses between the two facing boxes; the tracer
keeps as much of their fluid behind it as it had, so it goes over when the fluid does
and lands in the other volume's facing box. A joint's own tracers never leave it. The
joint and the volume beyond it are found by the tracer's volume's place (`places.py`).
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, wave_of
from physics.engine.axiom_3_structure.definitions.places import beside, stride_of
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_4_crossing.definitions.facing import faced
from physics.engine.axiom_4_crossing.definitions.passage import (
    passing_turn, row_minus, row_of, rub_share)
from physics.view.definitions.carried import carried

wp.set_module_options({"enable_backward": False})  # nothing here is ever run backward


@wp.kernel
def _carry_across(quanta: wp.array2d(dtype=wp.int32), phase: wp.array2d(dtype=wp.float32),
                  own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
                  tile_slots: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int32),
                  side: wp.int32, axis: wp.int32, cells: wp.int32, across: wp.int32,
                  row: wp.int32, tail: wp.array2d(dtype=wp.vec3)):
    tracer = wp.tid()
    place = tail[row, tracer]
    slot = wp.clamp(int(place[1]) / across, 0, cells - 1)
    place[1] -= float(slot * across)
    line, end = place[axis] - float(across - 1), 0  # along the pair, were this the volume before
    if line < 0.0:  # not in its cell's last box: in its first, the pair's volume after?
        line, end = line + float(across), 1
    volume = own_cell[slot]
    if line < 0.0 or line >= 2.0 or cell_quanta[volume] <= 0 or side == 0:
        return
    joint = beside(volume, side, axis, 1 - 2 * end)
    beyond = beside(volume, side, axis, 2 - 4 * end)
    if joint < 0 or beyond < 0 or cell_quanta[joint] >= 0:
        return
    one, two = (axis + 1) % 3, (axis + 2) % 3
    here = wp.vec2i(wp.clamp(int(place[one]), 0, across - 1), wp.clamp(int(place[two]), 0, across - 1))
    between, other = slot_at(tile_home, tile_slots, joint), slot_at(tile_home, tile_slots, beyond)
    if between < 0 or other < 0:
        return
    faces = faced(row_of(here[0], here[1], across, axis), across, axis)
    minus = row_minus(quanta, between, faces[1], stride_of(across, axis), across)
    slots = wp.vec2i(slot, other)
    if end == 1:
        slots = wp.vec2i(other, slot)
    low_has, high_has = quanta[slots[0], faces[0]], quanta[slots[1], faces[1]]
    if minus <= wp.float64(0.0) or low_has + high_has <= 0:
        return
    _low_wave, high_wave = pair_rotation(
        wave_of(low_has, phase[slots[0], faces[0]]), wave_of(high_has, phase[slots[1], faces[1]]),
        passing_turn(rub_share(minus, low_has, high_has), across))
    stays = float(low_has + high_has) - float(wp.dot(high_wave, high_wave))
    line = carried(line, float(low_has), float(high_has), stays)
    now = int(line >= 1.0)  # the volume it is in now: the one before, or over in the one after
    if now == 0:
        line = wp.min(line, 0.999) + float(across)  # never onto the next slot
    place[axis] = line - 1.0
    place[1] += float(slots[now] * across)
    tail[row, tracer] = place


def carry_across(store, axis: int, cells: int, tail: wp.array, row: int) -> None:
    """Carry the tracers of row `row` across the own joints that lie along `axis`."""
    wp.launch(_carry_across, dim=tail.shape[1], device=tail.device,
              inputs=[store.box_quanta, store.box_phase, store.own_cell, store.tile_home,
                      store.tile_slots, store.cell_quanta, store.side,
                      axis, cells, store.across, row, tail])
