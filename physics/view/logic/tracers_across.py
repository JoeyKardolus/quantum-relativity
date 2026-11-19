"""Tracers across: a tracer in a volume's box that faces a joint is carried across with the fluid.

The engine's own passage says how much crosses between the two facing boxes; the tracer
keeps as much of their fluid behind it as it had, so it goes over when the fluid does
and lands in the other volume's facing box. A joint's own tracers never leave it. The
joint is looked for among the own cells: the one that names the tracer's volume on that
side and whose part of the side holds the tracer's box.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, wave_of
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_4_crossing.definitions.facing import faced, leads, row_facing
from physics.engine.axiom_4_crossing.definitions.passage import (
    passing_turn, row_minus, row_of, rub_share, stride_of)
from physics.view.definitions.carried import carried

wp.set_module_options({"enable_backward": False})  # nothing here is ever run backward


@wp.kernel
def _carry_across(quanta: wp.array2d(dtype=wp.int32), phase: wp.array2d(dtype=wp.float32),
                  own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
                  tile_slots: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int32),
                  joint_ends: wp.array(dtype=wp.vec3i), joint_part: wp.array2d(dtype=wp.vec4),
                  axis: wp.int32, cells: wp.int32, across: wp.int32,
                  row: wp.int32, tail: wp.array2d(dtype=wp.vec3)):
    tracer = wp.tid()
    place = tail[row, tracer]
    slot = wp.clamp(int(place[1]) / across, 0, cells - 1)
    place[1] -= float(slot * across)
    line, end = place[axis] - float(across - 1), 0  # along the pair, were this the volume before
    if line < 0.0:  # not in its cell's last box: in its first, the pair's volume after?
        line, end = line + float(across), 1
    if line < 0.0 or line >= 2.0 or cell_quanta[own_cell[slot]] <= 0:
        return
    one, two = (axis + 1) % 3, (axis + 2) % 3
    here = wp.vec2i(wp.clamp(int(place[one]), 0, across - 1), wp.clamp(int(place[two]), 0, across - 1))
    for between in range(cells):
        joint = own_cell[between]
        if joint_ends[joint][2] != axis or joint_ends[joint][end] != own_cell[slot]:
            continue
        part = joint_part[joint, end]
        first = row_facing(part[0], part[1], here[0], across)
        third = row_facing(part[2], part[3], here[1], across)
        other = slot_at(tile_home, tile_slots, joint_ends[joint][1 - end])
        faces = faced(joint_part, joint, wp.max(first, 0), wp.max(third, 0), across, axis)
        minus = row_minus(quanta, between, row_of(wp.max(first, 0), wp.max(third, 0), across, axis),
                          stride_of(across, axis), across)
        if (first < 0 or third < 0 or other < 0 or wp.min(faces) < 0 or minus <= wp.float64(0.0)
                or leads(joint_part, joint, first, third, across) == 0):
            continue
        slots = wp.vec2i(slot, other)
        if end == 1:
            slots = wp.vec2i(other, slot)
        low_has, high_has = quanta[slots[0], faces[0]], quanta[slots[1], faces[1]]
        _low_wave, high_wave = pair_rotation(
            wave_of(low_has, phase[slots[0], faces[0]]), wave_of(high_has, phase[slots[1], faces[1]]),
            passing_turn(rub_share(minus, low_has, high_has), across))
        stays = float(low_has + high_has) - float(wp.dot(high_wave, high_wave))
        line = carried(line, float(low_has), float(high_has), stays)
        now = int(line >= 1.0)  # the volume it is in now: the one before, or over in the one after
        if now != end:  # over: into the other volume's facing box, as far into it as it was
            box = faces[now]
            place[one] += float(box / stride_of(across, one) % across - here[0])
            place[two] += float(box / stride_of(across, two) % across - here[1])
        if now == 0:
            line = wp.min(line, 0.999) + float(across)  # never onto the next slot
        place[axis] = line - 1.0
        place[1] += float(slots[now] * across)
        tail[row, tracer] = place
        return


def carry_across(store, axis: int, cells: int, tail: wp.array, row: int) -> None:
    """Carry the tracers of row `row` across the own joints that lie along `axis`."""
    wp.launch(_carry_across, dim=tail.shape[1], device=tail.device,
              inputs=[store.box_quanta, store.box_phase, store.own_cell, store.tile_home,
                      store.tile_slots, store.cell_quanta, store.joint_ends, store.joint_part,
                      axis, cells, store.across, row, tail])
