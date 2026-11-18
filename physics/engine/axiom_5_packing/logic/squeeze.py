"""Squeeze: the pull across every own joint and its push back, read off the cells' quanta.

One small launch over the own slots, before anything acts on a joint in a step, so every
box of a joint sees the same two numbers. A joint is squeezed while the pull is the bigger.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at, slot_of
from physics.engine.axiom_5_packing.definitions.pull_push import pull_across, push_back


@wp.kernel
def _squeeze(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
             tile_slots: wp.array(dtype=wp.int32), calm: wp.float64,
             cell_quanta: wp.array(dtype=wp.int32), own_squeeze: wp.array(dtype=wp.vec2d)):
    slot = wp.tid()
    joint = own_cell[slot]
    own_squeeze[slot] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    if joint % 2 == 0 or joint + 1 >= cell_quanta.shape[0]:
        return  # only a cell at an odd place has a volume on both sides
    if (slot_at(tile_home, tile_slots, joint - 1) < 0
            or slot_at(tile_home, tile_slots, joint + 1) < 0):
        return  # a calm volume does not press yet (layer 3)
    held = wp.float64(-cell_quanta[joint]) / calm
    pull = pull_across(wp.float64(cell_quanta[joint - 1]) / calm,
                       wp.float64(cell_quanta[joint + 1]) / calm, held)
    own_squeeze[slot] = wp.vec2d(pull, push_back(held))


def squeeze(store: Store) -> None:
    """Write each own joint's pull and push into `own_squeeze`."""
    wp.launch(_squeeze, dim=store.count, device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots,
                      wp.float64(QUANTA_A_BOX * store.across ** 3), store.cell_quanta,
                      store.own_squeeze])


def pull_push(store: Store, joint: int) -> np.ndarray:
    """The pull across an own joint and its push back as they are now, read on the host."""
    squeeze(store)
    return store.own_squeeze.numpy()[slot_of(store, joint)]
