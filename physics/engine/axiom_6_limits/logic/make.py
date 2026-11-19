"""Make: a squeezed joint makes more minus quanta out of itself, so it shrinks.

While the pull across a joint beats its push, every box of the joint makes whole quanta
out of its own: what the joint gives in a step (`given`) times how hard the volumes'
fluid presses on the box's row (`pressed`). What is made is booked in `cell_made`;
nothing is ever unmade, and the volumes lose nothing. Counted a turn of a swirl, not a
step: boxes are not space.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.whole_move import odd_part
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_4_crossing.definitions.facing import faced
from physics.engine.axiom_3_structure.definitions.merged import joint_volumes
from physics.engine.axiom_5_packing.definitions.pull_push import given, pressed


@wp.kernel
def _make(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
          tile_slots: wp.array(dtype=wp.int32), own_whole: wp.array(dtype=wp.int32),
          side: wp.int32, across: wp.int32,
          a_turn: wp.float64,
          own_squeeze: wp.array(dtype=wp.vec4d),
          cell_quanta: wp.array(dtype=wp.int32), cell_made: wp.array(dtype=wp.int32),
          box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, layer, around, along = wp.tid()
    gives = given(own_squeeze[slot], a_turn)
    if gives <= wp.float64(0.0):
        return  # not squeezed: the joint's energy stays as it is
    joint, axis = own_cell[slot], wp.int32(own_squeeze[slot][3])
    box = (layer * across + around) * across + along
    row = around % (across / 6) * across + along  # the caps' row this box lies behind
    ends = joint_volumes(joint, side, axis, tile_home, tile_slots, own_whole, own_cell)
    share = gives * pressed(box_quanta, cell_quanta, tile_home, tile_slots, ends,
                            faced(row, across, axis))
    made = wp.int32(wp.floor(wp.float64(-box_quanta[slot, box]) * share
                             + odd_part(box_phase[slot, box], across)))
    box_quanta[slot, box] -= made
    wp.atomic_add(cell_quanta, joint, -made)
    wp.atomic_add(cell_made, joint, made)


def make(store: Store) -> None:
    """One step of making in every own joint that `squeeze` found squeezed."""
    across = store.across
    wp.launch(_make, dim=(store.count, across, across, across), device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.own_whole,
                      store.side, across, wp.float64(store.turn_per_step / (2.0 * wp.pi)),
                      store.own_squeeze, store.cell_quanta, store.cell_made, store.box_quanta,
                      store.box_phase])
