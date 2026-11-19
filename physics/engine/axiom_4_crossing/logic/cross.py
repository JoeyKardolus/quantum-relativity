"""Cross: one step of plus energy crossing every joint that lies between two volumes.

A volume's fluid has one common flow a direction (`cell_flow`: its quanta times how fast
they go). Energy has no identity, so two flows that meet in a joint are set against each
other: only what the giver's flow toward the joint has over the taker's crosses; the rest
presses, and its flow is handed on through the joint. The share that goes a step is the
speed, never past the top speed, over the distance to the next volume (`apart_of`), drawn
from every box of the giver alike and at most a third of a box (a box's fluid can go three
directions at once); it lands box for box in the taker's mirror, still moving. The joint's
friction holds back the rubbing share (`rub_share`), which stays. Two volumes that only hold
different amounts exchange nothing; out of a volume past the limit nothing crosses. Quanta
and flow are booked to the whole number, read off copies made before the launch.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.whole_move import odd_part
from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import joint_axis, joint_ends
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_4_crossing.definitions.passage import row_minus, rub_share
from physics.engine.axiom_5_packing.definitions.pull_push import apart_of
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP, TOP_SPEED

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _cross(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
           tile_slots: wp.array(dtype=wp.int32), side: wp.int32, across: wp.int32,
           a_turn: wp.float64, calm: wp.float64, deep: wp.float32, top: wp.float64,
           cell_pull: wp.array(dtype=wp.float32), held: wp.array(dtype=wp.int64),
           flowed: wp.array2d(dtype=wp.int64), had: wp.array2d(dtype=wp.int32),
           cell_quanta: wp.array(dtype=wp.int64), cell_flow: wp.array2d(dtype=wp.int64),
           box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, box, way = wp.tid()  # way 0: from the volume before to the one after; 1: back
    joint, axis = own_cell[slot], joint_axis(own_cell[slot], side)
    if axis < 0 or held[joint] >= wp.int64(0):
        return  # only a joint lets energy cross
    ends = joint_ends(joint, side, axis)
    giver, taker = ends[way], ends[1 - way]
    gives, takes = slot_at(tile_home, tile_slots, giver), slot_at(tile_home, tile_slots, taker)
    minus = row_minus(had, slot, box % across, across, axis)
    if gives < 0 or takes < 0 or minus <= wp.float64(0.0) or cell_pull[giver] >= deep or wp.min(held[giver], held[taker]) <= wp.int64(0):
        return  # no volume either side, no minus between, or past the limit: nothing passes
    own = wp.float64(flowed[giver, axis]) * wp.float64(1 - 2 * way)  # the giver's flow toward this joint
    if own <= wp.float64(0.0):
        return  # only fluid that goes this way presses or crosses
    over = own - wp.max(wp.float64(flowed[taker, axis]) * wp.float64(2 * way - 1), wp.float64(0.0))
    apart = apart_of(wp.float64(held[ends[0]]) / calm, wp.float64(held[ends[1]]) / calm,
                     -wp.float64(held[joint]) / calm)
    back = box + (across - 1 - 2 * (box / across % across)) * across  # the mirror round the core
    has, third = had[gives, box], wp.float64(1.0 / 3.0)
    reach = wp.float64(has) * wp.min(wp.min(own / wp.float64(held[giver]), top) * a_turn / apart, third)
    goes = wp.float64(has) * wp.min(wp.min(over / wp.float64(held[giver]), top) * a_turn / apart, third)
    rub, part = rub_share(minus, has, had[takes, back]), odd_part(box_phase[gives, box], across)
    moved = wp.clamp(wp.int32(wp.floor(goes * (wp.float64(1.0) - rub) + part)), 0, has / 3)
    if over <= wp.float64(0.0):
        moved = 0  # met by as much coming the other way: it only presses
    wp.atomic_sub(box_quanta, gives, box, moved)
    wp.atomic_add(box_quanta, takes, back, moved)
    wp.atomic_sub(cell_quanta, giver, wp.int64(moved))
    wp.atomic_add(cell_quanta, taker, wp.int64(moved))
    for along in range(3):  # what crosses takes its share of the flow, every direction of it
        share = wp.float64(flowed[giver, along]) / wp.float64(held[giver])
        carried = wp.int64(share * wp.float64(moved))
        if along == axis:  # what is held back presses on the joint: its flow goes on, it stays
            carried = wp.int64(share * wp.max(reach, wp.float64(moved)))
        wp.atomic_sub(cell_flow, giver, along, carried)
        wp.atomic_add(cell_flow, taker, along, carried)


def cross(store) -> None:
    """One step of crossing at every own joint, both ways."""
    across = store.across
    wp.launch(_cross, dim=(store.count, across ** 3, 2), device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.side, across,
                      wp.float64(store.turn_per_step / (2.0 * wp.pi)),
                      wp.float64(QUANTA_A_BOX * across ** 3), DEEP, wp.float64(TOP_SPEED), store.cell_pull,
                      wp.clone(store.cell_quanta), wp.clone(store.cell_flow), wp.clone(store.box_quanta),
                      store.cell_quanta, store.cell_flow, store.box_quanta, store.box_phase])
