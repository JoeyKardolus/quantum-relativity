"""Merged: which volume a place is part of once joints have collapsed.

A collapse makes one volume of two and the joint between (`collapse.py`). The places keep
their numbers, so each own slot names the slot of the volume it is part of (`own_whole`:
itself until a collapse), and a joint's two volumes are found through it. The slots of one
volume form a ring (`own_ring`). A merged volume is one ball with six sides, so the joints
that now lie on one side of it share that side: `own_part` holds a joint's rank among them
and their count, for the volume before it and for the one after (`ALONE`: by itself).
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import joint_ends
from physics.engine.axiom_3_structure.definitions.store import TILE, slot_at

ALONE = np.array([0, 1, 0, 1], dtype=np.int32)  # a joint that shares neither volume's side


@wp.func
def whole_of(tile_home: wp.array(dtype=wp.int32), tile_slots: wp.array(dtype=wp.int32),
             own_whole: wp.array(dtype=wp.int32), own_cell: wp.array(dtype=wp.int32),
             cell: wp.int32) -> wp.int32:
    """The place of the volume that the place `cell` is part of: itself unless merged."""
    slot = slot_at(tile_home, tile_slots, cell)
    if slot < 0:
        return cell
    return own_cell[own_whole[slot]]


@wp.func
def joint_volumes(joint: wp.int32, side: wp.int32, axis: wp.int32,
                  tile_home: wp.array(dtype=wp.int32), tile_slots: wp.array(dtype=wp.int32),
                  own_whole: wp.array(dtype=wp.int32), own_cell: wp.array(dtype=wp.int32)) -> wp.vec2i:
    """A joint's two volumes as they are now: the one before it along its axis, the one after."""
    ends = joint_ends(joint, side, axis)
    return wp.vec2i(whole_of(tile_home, tile_slots, own_whole, own_cell, ends[0]),
                    whole_of(tile_home, tile_slots, own_whole, own_cell, ends[1]))


@wp.func
def allotted(part: wp.vec4i, row: wp.int32) -> bool:
    """Whether a row of a volume's side is this joint's: the joints on a side take its rows in turn."""
    return row % part[1] == part[0] and row % part[3] == part[2]


def slots_of(store, cells) -> np.ndarray:
    """On the host: the slots of own cells, by their numbers."""
    cells = np.asarray(cells)
    return store.tile_slots.numpy()[store.tile_home.numpy()[cells // TILE] * TILE + cells % TILE]
