"""Tracers across: a tracer in a volume's box that faces a joint is carried across with the fluid.

The engine's own passage says how much crosses between the two facing boxes; the tracer
keeps as much of their fluid behind it as it had, so it goes over when the fluid does
and lands in the other volume. A joint's own tracers never leave it.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, wave_of
from physics.engine.axiom_3_structure.definitions.grid import cell_beside, volume_beside
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_4_crossing.definitions.passage import (
    passing_turn, row_minus, row_of, rub_share, stride_of)
from physics.view.definitions.carried import carried

wp.set_module_options({"enable_backward": False})  # nothing here is ever run backward


@wp.kernel
def _carry_across(quanta: wp.array2d(dtype=wp.int32), phase: wp.array2d(dtype=wp.float32),
                  own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
                  tile_slots: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int32),
                  grid: wp.vec3i, axis: wp.int32, cells: wp.int32, across: wp.int32,
                  row: wp.int32, tail: wp.array2d(dtype=wp.vec3)):
    tracer = wp.tid()
    place = tail[row, tracer]
    slot = wp.clamp(int(place[1]) / across, 0, cells - 1)
    place[1] -= float(slot * across)
    line, way = place[axis] - float(across - 1), 1  # along the pair, were this the volume before
    if line < 0.0:  # not in its cell's last box: in its first, the pair's volume after?
        line, way = line + float(across), -1
    joint = cell_beside(grid, own_cell[slot], axis, way)
    if line < 0.0 or line >= 2.0 or joint < 0 or cell_quanta[own_cell[slot]] <= 0:
        return
    other = volume_beside(cell_quanta, grid, joint, axis, way)
    if other < 0 or slot_at(tile_home, tile_slots, other) < 0:
        return
    low, high = slot, slot_at(tile_home, tile_slots, other)
    if way < 0:
        low, high = high, slot
    stride = stride_of(across, axis)
    first = row_of(wp.clamp(int(place[1 - axis]), 0, across - 1),
                   wp.clamp(int(place[2]), 0, across - 1), across, axis)
    last = first + (across - 1) * stride
    minus = row_minus(quanta, slot_at(tile_home, tile_slots, joint), first, stride, across)
    if minus <= wp.float64(0.0):
        return
    low_has, high_has = quanta[low, last], quanta[high, first]
    _low_wave, high_wave = pair_rotation(
        wave_of(low_has, phase[low, last]), wave_of(high_has, phase[high, first]),
        passing_turn(rub_share(minus, low_has, high_has), across))
    stays = float(low_has + high_has) - float(wp.dot(high_wave, high_wave))
    line = carried(line, float(low_has), float(high_has), stays)
    slot = high  # over, in the volume after; or still in the one before
    if line < 1.0:
        slot, line = low, line + float(across)
    place[axis] = line - 1.0
    place[1] += float(slot * across)
    tail[row, tracer] = place


def carry_across(store, axis: int, cells: int, tail: wp.array, row: int) -> None:
    """Carry the tracers of row `row` across the own joints that lie along `axis`."""
    wp.launch(_carry_across, dim=tail.shape[1], device=tail.device,
              inputs=[store.box_quanta, store.box_phase, store.own_cell, store.tile_home,
                      store.tile_slots, store.cell_quanta, wp.vec3i(*store.grid), axis, cells,
                      store.across, row, tail])
