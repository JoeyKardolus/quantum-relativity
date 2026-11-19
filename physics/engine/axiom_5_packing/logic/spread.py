"""Spread: one step of the pull being handed on, one neighbour and no further.

One launch over the own cells. Every cell reads its neighbours' pull off a copy made before
the launch, so the pull moves one cell a step whatever the order: a joint takes the mean of
its two volumes', a volume its own plus energy and a sixth of each side's joint's. A merged
volume takes its joints at all its places, those that share a side sharing its sixth. A calm
cell and a free place hold no pull. A store that keeps no places has no neighbours: no pull.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.merged import joint_volumes
from physics.engine.axiom_3_structure.definitions.places import beside, joint_axis
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at
from physics.engine.axiom_5_packing.definitions.pull import SIDES

wp.set_module_options({"enable_backward": False})  # nothing here is ever run backward


@wp.func
def _handed(place: wp.int32, side: wp.int32, tile_home: wp.array(dtype=wp.int32),
            tile_slots: wp.array(dtype=wp.int32), own_part: wp.array(dtype=wp.vec4i),
            cell_quanta: wp.array(dtype=wp.int32), was: wp.array(dtype=wp.float32)) -> float:
    """The pull of the live joints at one place's six sides, each over the joints that share its side."""
    handed = float(0.0)
    for axis in range(3):
        for way in range(-1, 2, 2):
            joint = beside(place, side, axis, way)
            if joint >= 0 and cell_quanta[joint] < 0 and slot_at(tile_home, tile_slots, joint) >= 0:
                shared = own_part[slot_at(tile_home, tile_slots, joint)][2 - way]
                handed += was[joint] / float(shared)
    return handed


@wp.kernel
def _spread(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
            tile_slots: wp.array(dtype=wp.int32), own_whole: wp.array(dtype=wp.int32),
            own_ring: wp.array(dtype=wp.int32), own_part: wp.array(dtype=wp.vec4i),
            side: wp.int32, calm: wp.float32, cell_quanta: wp.array(dtype=wp.int32),
            was: wp.array(dtype=wp.float32), cell_pull: wp.array(dtype=wp.float32)):
    slot = wp.tid()
    cell = own_cell[slot]
    along = joint_axis(cell, side)
    if along >= 0:  # a joint's place: a joint carries the pull of its two volumes, adds none
        ends = joint_volumes(cell, side, along, tile_home, tile_slots, own_whole, own_cell)
        cell_pull[cell] = 0.5 * (was[ends[0]] + was[ends[1]]) * float(cell_quanta[cell] < 0)
        return
    if own_whole[slot] != slot:
        cell_pull[cell] = 0.0  # part of a merged volume: the slot the volume stands in answers
        return
    handed = _handed(cell, side, tile_home, tile_slots, own_part, cell_quanta, was)
    part = int(own_ring[slot])
    while part != slot:
        handed += _handed(own_cell[part], side, tile_home, tile_slots, own_part, cell_quanta, was)
        part = own_ring[part]
    cell_pull[cell] = float(cell_quanta[cell]) / calm + handed / SIDES


def spread(store: Store) -> None:
    """Hand every own cell's pull on by one neighbour."""
    if store.side == 0:
        return
    wp.launch(_spread, dim=store.count, device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.own_whole,
                      store.own_ring, store.own_part, store.side,
                      float(QUANTA_A_BOX * store.across ** 3),
                      store.cell_quanta, wp.clone(store.cell_pull), store.cell_pull])
