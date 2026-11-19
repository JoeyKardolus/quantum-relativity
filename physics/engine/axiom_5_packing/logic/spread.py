"""Spread: one step of the pull being handed on, one neighbour and no further.

One launch over the own cells. Every cell reads its neighbours' pull off a copy made before
the launch, so the pull moves one cell a step whatever the order: a joint takes the mean of
its two volumes', a volume its own plus energy and a sixth of each side's joint's. A joint
that has collapsed holds plus: it carries the pull on as before and adds its own energy. A
calm cell and a free place hold no pull. A store that keeps no places has no neighbours.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import beside, joint_axis, joint_ends
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at
from physics.engine.axiom_5_packing.definitions.pull import SIDES

wp.set_module_options({"enable_backward": False})  # nothing here is ever run backward


@wp.func
def _handed(place: wp.int32, side: wp.int32, tile_home: wp.array(dtype=wp.int32),
            tile_slots: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int32),
            was: wp.array(dtype=wp.float32)) -> float:
    """The pull of the cells at a volume's six sides: its joints, standing or collapsed."""
    handed = float(0.0)
    for axis in range(3):
        for way in range(-1, 2, 2):
            joint = beside(place, side, axis, way)
            if joint >= 0 and cell_quanta[joint] != 0 and slot_at(tile_home, tile_slots, joint) >= 0:
                handed += was[joint]
    return handed


@wp.kernel
def _spread(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
            tile_slots: wp.array(dtype=wp.int32), side: wp.int32, calm: wp.float32,
            cell_quanta: wp.array(dtype=wp.int32), was: wp.array(dtype=wp.float32),
            cell_pull: wp.array(dtype=wp.float32)):
    cell = own_cell[wp.tid()]
    along = joint_axis(cell, side)
    own = wp.max(float(cell_quanta[cell]), 0.0) / calm  # plus pulls; a joint's minus adds none
    if along >= 0:  # a joint's place: it carries the mean of its two volumes' pull
        ends = joint_ends(cell, side, along)
        cell_pull[cell] = (own + 0.5 * (was[ends[0]] + was[ends[1]])) * float(cell_quanta[cell] != 0)
        return
    cell_pull[cell] = own + _handed(cell, side, tile_home, tile_slots, cell_quanta, was) / SIDES


def spread(store: Store) -> None:
    """Hand every own cell's pull on by one neighbour."""
    if store.side == 0:
        return
    wp.launch(_spread, dim=store.count, device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.side,
                      float(QUANTA_A_BOX * store.across ** 3),
                      store.cell_quanta, wp.clone(store.cell_pull), store.cell_pull])
