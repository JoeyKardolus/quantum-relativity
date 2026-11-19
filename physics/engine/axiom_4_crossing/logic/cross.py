"""Cross: one step of plus energy crossing every joint that lies between two volumes.

What moves energy across a joint, two lines. A volume's fluid has one common flow a
direction (`cell_flow`: its quanta times how fast they go). The fluid that goes toward a
joint leaves from the outside of the ring whirl that points at it, once round its core,
takes its share of the flow along and arrives round the core of the next volume's ring,
box for box in the mirror, still moving the same way: what one volume loses of quanta and
of flow the other gains, to the whole number.

How much tries a step: the flow over the distance to the next volume (`apart_of`), never
more than those boxes hold. The joint's friction holds back the rubbing share
(`rub_share`); that stays in the volume it came from and stops there, so the flow toward
the joint is less by its share. Two volumes that only hold different amounts exchange
nothing here (the leak is its own thing: `leak.py`). Out of a volume past the limit nothing
crosses. Quanta, flow and boxes are read off copies made before the launch, so a run repeats.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.whole_move import odd_part
from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import joint_axis, joint_ends
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_2_swirls.definitions.ring_boxes import box_at
from physics.engine.axiom_4_crossing.definitions.passage import row_minus, rub_share
from physics.engine.axiom_5_packing.definitions.pull_push import apart_of
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _cross(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
           tile_slots: wp.array(dtype=wp.int32), side: wp.int32, across: wp.int32,
           a_turn: wp.float64, calm: wp.float64, deep: wp.float32,
           cell_pull: wp.array(dtype=wp.float32), held: wp.array(dtype=wp.int64),
           flowed: wp.array2d(dtype=wp.int64), had: wp.array2d(dtype=wp.int32),
           cell_quanta: wp.array(dtype=wp.int64), cell_flow: wp.array2d(dtype=wp.int64),
           box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, row, around, way = wp.tid()  # way 0: from the volume before to the one after; 1: back
    joint, axis = own_cell[slot], joint_axis(own_cell[slot], side)
    if axis < 0 or held[joint] >= wp.int64(0):
        return  # only a joint lets energy cross
    ends, last = joint_ends(joint, side, axis), across / 3 - 1  # the rings' outside layer
    out, back = box_at(across, axis, last, around, row), box_at(across, axis, last, across - 1 - around, row)
    giver, taker = ends[way], ends[1 - way]
    gives, takes = slot_at(tile_home, tile_slots, giver), slot_at(tile_home, tile_slots, taker)
    minus = row_minus(had, slot, row, across, axis)
    if gives < 0 or takes < 0 or minus <= wp.float64(0.0) or cell_pull[giver] >= deep:
        return  # no volume either side, no minus between, or past the limit: nothing passes
    toward = wp.float64(flowed[giver, axis]) * wp.float64(1 - 2 * way)  # the flow toward this joint
    if toward <= wp.float64(0.0) or held[giver] <= wp.int64(0) or held[taker] <= wp.int64(0):
        return  # only fluid that goes this way crosses
    has = had[gives, out]
    apart = apart_of(wp.float64(held[ends[0]]) / calm, wp.float64(held[ends[1]]) / calm,
                     -wp.float64(held[joint]) / calm)
    reach = wp.min(toward * a_turn / apart, wp.float64(held[giver])) / wp.float64(across * across)
    rub, part = rub_share(minus, has, has), odd_part(box_phase[gives, out], across)
    moved = wp.clamp(wp.int32(wp.floor(reach * (wp.float64(1.0) - rub) + part)), 0, has)
    wp.atomic_sub(box_quanta, gives, out, moved)
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
    wp.launch(_cross, dim=(store.count, across, across, 2), device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.side, across,
                      wp.float64(store.turn_per_step / (2.0 * wp.pi)),
                      wp.float64(QUANTA_A_BOX * across ** 3), DEEP, store.cell_pull,
                      wp.clone(store.cell_quanta), wp.clone(store.cell_flow), wp.clone(store.box_quanta),
                      store.cell_quanta, store.cell_flow, store.box_quanta, store.box_phase])
