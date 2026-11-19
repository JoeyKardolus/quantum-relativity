"""Cross: one step of plus energy crossing every joint that lies between two volumes.

A joint names its axis and its two volumes (`joint_ends`). Each box of the one volume that
faces a row of the joint moves with the box of the other facing it (`facing.py`), by the
pair rotation at the joint's `passing_turn`: energy goes the way its phase carries it and
arrives with its phase, so it is still moving. Friction then takes its share of the motion
across; what it holds back stays in the volume it came from. What one volume loses the
other gains; the joint's quanta stay. One launch an axis, so a volume's corner box is never
in two pairs of one launch.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, phase_of, wave_of
from physics.engine.axiom_1_energy.definitions.whole_move import whole_move
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_4_crossing.definitions.facing import faced, leads
from physics.engine.axiom_4_crossing.definitions.passage import (
    layer_part, less_step, passing_turn, row_minus, row_of, rub_share, stride_of)

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _cross(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
           tile_slots: wp.array(dtype=wp.int32), joint_ends: wp.array(dtype=wp.vec3i),
           joint_part: wp.array2d(dtype=wp.vec4), axis: wp.int32,
           across: wp.int32, seed: wp.uint32, step: wp.uint32,
           cell_quanta: wp.array(dtype=wp.int32),
           box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, first, third = wp.tid()
    joint, stride = own_cell[slot], stride_of(across, axis)
    row = row_of(first, third, across, axis)  # the row's first box, facing the volume before
    before, after = joint_ends[joint][0], joint_ends[joint][1]
    if joint_ends[joint][2] != axis or leads(joint_part, joint, first, third, across) == 0:
        return  # only a joint that lies along this axis lets energy cross, one row a box
    faces = faced(joint_part, joint, first, third, across, axis)
    left, right = slot_at(tile_home, tile_slots, before), slot_at(tile_home, tile_slots, after)
    minus = row_minus(box_quanta, slot, row, stride, across)
    if left < 0 or right < 0 or minus <= wp.float64(0.0) or wp.min(faces) < 0:
        return  # a calm volume is not crossed into yet; with no minus between, nothing passes
    low, high = faces[0], faces[1]  # the two volumes' boxes that face this row
    low_has = box_quanta[left, low]
    high_has = box_quanta[right, high]
    rub = rub_share(minus, low_has, high_has)
    low_wave, high_wave = pair_rotation(wave_of(low_has, box_phase[left, low]),
                                        wave_of(high_has, box_phase[right, high]),
                                        passing_turn(rub, across))
    box = wp.uint64(joint) * wp.uint64(box_quanta.shape[1]) + wp.uint64(row)
    flow = wp.dot(high_wave, high_wave) - wp.float64(high_has)  # what the second volume gains
    moved = whole_move(flow, draw_uniform(seed, step, wp.uint32(6), box), low_has, high_has)
    step_across = less_step(phase_of(low_wave, box_phase[left, low]),
                            phase_of(high_wave, box_phase[right, high]), rub * layer_part(across))
    box_quanta[left, low] = low_has - moved
    box_quanta[right, high] = high_has + moved
    box_phase[left, low] = step_across[0]
    box_phase[right, high] = step_across[1]
    wp.atomic_add(cell_quanta, before, -moved)
    wp.atomic_add(cell_quanta, after, moved)


def cross(store, before=None) -> None:
    """One step of crossing at every own joint; `before(axis)` lets a watcher look first."""
    for axis in range(3):
        if before:
            before(axis)
        wp.launch(_cross, dim=(store.count, store.across, store.across),
                  device=store.box_quanta.device,
                  inputs=[store.own_cell, store.tile_home, store.tile_slots,
                          store.joint_ends, store.joint_part, axis, store.across, store.seed,
                          store.step, store.cell_quanta, store.box_quanta, store.box_phase])
