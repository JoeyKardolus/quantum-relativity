"""Cross: one step of plus energy crossing every joint that lies between two volumes.

A joint's place gives its axis and its two volumes (`places.py`). Each box of the one volume that
faces a row of the joint moves with the box of the other facing it (`facing.py`), by the
pair rotation at the joint's `passing_turn`, the first wave turned on by the joint's own
whirl between its two caps (`gear`): energy goes the way its phase carries it and arrives
with its phase, so it is still moving, and in step with the whirl it arrives in. The pull turns the two phases first (`bent`),
so more crosses toward the deeper pull. Friction then takes its share of the motion
across; what it holds back stays in the volume it came from. What one volume loses the
other gains; the joint's quanta stay. Out of a volume past the limit nothing crosses. One
launch an axis, so a volume's corner box is never in two pairs of one launch.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, phase_of, wave_of
from physics.engine.axiom_1_energy.definitions.whole_move import odd_part, whole_move
from physics.engine.axiom_3_structure.definitions.places import joint_axis, joint_ends
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_4_crossing.definitions.facing import faced, gear
from physics.engine.axiom_4_crossing.definitions.passage import (
    layer_part, less_step, passing_turn, row_minus, rub_share)
from physics.engine.axiom_5_packing.definitions.pull import bent, differing
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _cross(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
           tile_slots: wp.array(dtype=wp.int32), side: wp.int32, axis: wp.int32,
           across: wp.int32, a_turn: wp.float64, deep: wp.float32,
           cell_pull: wp.array(dtype=wp.float32), own_turned: wp.array(dtype=wp.float32),
           cell_quanta: wp.array(dtype=wp.int32),
           box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, first, third = wp.tid()
    joint, row = own_cell[slot], first * across + third
    if cell_quanta[joint] >= 0 or joint_axis(joint, side) != axis:
        return  # only a joint that lies along this axis lets energy cross
    ends = joint_ends(joint, side, axis)
    before, after, faces = ends[0], ends[1], faced(row, across, axis)
    left, right = slot_at(tile_home, tile_slots, before), slot_at(tile_home, tile_slots, after)
    minus = row_minus(box_quanta, slot, row, across, axis)
    if left < 0 or right < 0 or minus <= wp.float64(0.0):
        return  # no volume either side, or no minus between: nothing passes
    low, high = faces[0], faces[1]  # the two volumes' boxes that face this row
    low_has, high_has = box_quanta[left, low], box_quanta[right, high]
    rub = rub_share(minus, low_has, high_has)
    geared = gear(box_phase, slot, faces) + own_turned[right] - own_turned[left]  # see `turn.py`
    turned = bent(box_phase[left, low] + geared, box_phase[right, high],
                  differing(cell_pull[before], cell_pull[after]), a_turn / layer_part(across))
    low_wave, high_wave = pair_rotation(wave_of(low_has, turned[0]), wave_of(high_has, turned[1]),
                                        passing_turn(rub, across))
    flow = wp.dot(high_wave, high_wave) - wp.float64(high_has)  # what the second volume gains
    moved = whole_move(flow, odd_part(box_phase[left, low], across), low_has, high_has)
    if cell_pull[before] >= deep:
        moved = wp.min(moved, 0)  # past the limit: nothing leaves the volume before,
    if cell_pull[after] >= deep:
        moved = wp.max(moved, 0)  # nor the volume after
    step_across = less_step(phase_of(low_wave, turned[0]), phase_of(high_wave, turned[1]),
                            rub * layer_part(across))
    box_quanta[left, low] = low_has - moved
    box_quanta[right, high] = high_has + moved
    box_phase[left, low] = less_step(step_across[0] - geared, 0.0, wp.float64(0.0))[0]
    box_phase[right, high] = step_across[1]
    wp.atomic_add(cell_quanta, before, -moved)
    wp.atomic_add(cell_quanta, after, moved)


def cross(store, before=None) -> None:
    """One step of crossing at every own joint; `before(axis)` lets a watcher look first."""
    for axis in range(3):
        if before:
            before(axis)
        wp.launch(_cross, dim=(store.count, store.across // 6, store.across),
                  device=store.box_quanta.device,
                  inputs=[store.own_cell, store.tile_home, store.tile_slots,
                          store.side, axis, store.across,
                          wp.float64(store.turn_per_step / (2.0 * wp.pi)), DEEP,
                          store.cell_pull, store.own_turned, store.cell_quanta, store.box_quanta,
                          store.box_phase])
