"""Squeeze: the pull across every own joint and its push back, read off the cells' quanta.

One small launch over the own slots, before anything acts on a joint in a step, so every
box of a joint sees the same numbers: the pull, the push, how far apart its two volumes
stand and the axis they stand along. A joint is squeezed while the pull is the bigger.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import joint_axis, joint_ends
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at, slot_of
from physics.engine.axiom_5_packing.definitions.pull_push import (
    apart_of, pull_across, push_back)

wp.set_module_options({"enable_backward": False})  # nothing here is ever run backward


@wp.kernel
def _squeeze(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
             tile_slots: wp.array(dtype=wp.int32), side: wp.int32,
             calm: wp.float64,
             cell_quanta: wp.array(dtype=wp.int32), own_squeeze: wp.array(dtype=wp.vec4d)):
    joint = own_cell[wp.tid()]
    slot, none = wp.tid(), wp.float64(0.0)
    own_squeeze[slot] = wp.vec4d(none, none, none, none)
    axis = joint_axis(joint, side)
    if axis < 0 or cell_quanta[joint] >= 0:
        return  # not a joint
    before, after = joint_ends(joint, side, axis)[0], joint_ends(joint, side, axis)[1]
    if cell_quanta[before] <= 0 or cell_quanta[after] <= 0:
        return  # only a joint between two volumes is pulled across
    if slot_at(tile_home, tile_slots, before) < 0 or slot_at(tile_home, tile_slots, after) < 0:
        return  # a calm volume does not press yet
    held = wp.float64(-cell_quanta[joint]) / calm
    left, right = wp.float64(cell_quanta[before]) / calm, wp.float64(cell_quanta[after]) / calm
    if held > none:
        own_squeeze[slot] = wp.vec4d(pull_across(left, right, held), push_back(held),
                                     apart_of(left, right, held), wp.float64(axis))


def squeeze(store: Store) -> None:
    """Write each own joint's pull and push into `own_squeeze`."""
    wp.launch(_squeeze, dim=store.count, device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.side,
                      wp.float64(QUANTA_A_BOX * store.across ** 3), store.cell_quanta,
                      store.own_squeeze])


def pull_push(store: Store, joint: int) -> np.ndarray:
    """The pull across an own joint and its push back as they are now, read on the host."""
    squeeze(store)
    return store.own_squeeze.numpy()[slot_of(store, joint), :2]
