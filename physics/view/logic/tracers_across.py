"""Tracers across: a tracer in a volume's cap box that faces a joint is carried across with the fluid.

The engine's own passage says how much crosses between the two facing cap boxes, the
first wave geared by the joint's whirl as the crossing gears it; the tracer keeps as much
of their fluid behind it as it had, so it goes over when the fluid does and lands in the
facing cap box of the volume beyond. Along the pair it runs out through the layers of the
one cap and in through the other's. A joint's own tracers never leave it.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, wave_of
from physics.engine.axiom_3_structure.definitions.places import beside
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_4_crossing.definitions.facing import faced, gear
from physics.engine.axiom_4_crossing.definitions.passage import passing_turn, row_minus, rub_share
from physics.view.definitions.carried import carried

wp.set_module_options({"enable_backward": False})  # nothing here is ever run backward


@wp.kernel
def _carry_across(quanta: wp.array2d(dtype=wp.int32), phase: wp.array2d(dtype=wp.float32),
                  own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
                  tile_slots: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
                  own_turned: wp.array(dtype=wp.float32), side: wp.int32, axis: wp.int32,
                  cells: wp.int32, across: wp.int32, row: wp.int32, tail: wp.array2d(dtype=wp.vec3)):
    tracer = wp.tid()
    place = tail[row, tracer]
    slot = wp.clamp(int(place[1]) / across, 0, cells - 1)
    around, layer = int(place[1]) - slot * across, int(place[0])
    end, front = 0, around  # in the front cap: this is the volume before the joint
    if around >= across / 2:
        end, front = 1, across - 1 - around  # in the back cap: the volume after it
    volume = own_cell[slot]
    if (layer != (axis + 1) * (across / 3) - 1 or front < across / 6 or front >= across / 3
            or cell_quanta[volume] <= wp.int64(0) or side == 0):
        return
    joint, beyond = beside(volume, side, axis, 1 - 2 * end), beside(volume, side, axis, 2 - 4 * end)
    if joint < 0 or beyond < 0 or cell_quanta[joint] >= wp.int64(0):
        return
    between, other = slot_at(tile_home, tile_slots, joint), slot_at(tile_home, tile_slots, beyond)
    if between < 0 or other < 0:
        return
    rows = (front - across / 6) * across + wp.clamp(int(place[2]), 0, across - 1)
    faces = faced(rows, across, axis)
    minus = row_minus(quanta, between, rows, across, axis)
    slots = wp.vec2i(slot, other)
    if end == 1:
        slots = wp.vec2i(other, slot)
    low_has, high_has = quanta[slots[0], faces[0]], quanta[slots[1], faces[1]]
    if minus <= wp.float64(0.0) or low_has + high_has <= 0:
        return
    geared = gear(phase, between, faces) + own_turned[slots[1]] - own_turned[slots[0]]
    _low_wave, high_wave = pair_rotation(
        wave_of(low_has, phase[slots[0], faces[0]] + geared), wave_of(high_has, phase[slots[1], faces[1]]),
        passing_turn(rub_share(minus, low_has, high_has), across))
    stays = float(low_has + high_has) - float(wp.dot(high_wave, high_wave))
    deep = place[0] - float(layer)  # out through the one cap's layer, in through the other's
    line = carried(deep + float(end) * (2.0 - 2.0 * deep), float(low_has), float(high_has), stays)
    if int(line >= 1.0) == end:
        return  # it stays in the volume it is in
    place[0] = float(layer) + 1.0 - wp.clamp(wp.abs(line - 1.0), 0.001, 1.0)
    place[1] = float(slots[1 - end] * across) + float(across) - (place[1] - float(slot * across))
    tail[row, tracer] = place


def carry_across(store, axis: int, cells: int, tail: wp.array, row: int) -> None:
    """Carry the tracers of row `row` across the own joints that lie along `axis`."""
    wp.launch(_carry_across, dim=tail.shape[1], device=tail.device,
              inputs=[store.box_quanta, store.box_phase, store.own_cell, store.tile_home,
                      store.tile_slots, store.cell_quanta, store.own_turned, store.side,
                      axis, cells, store.across, row, tail])
