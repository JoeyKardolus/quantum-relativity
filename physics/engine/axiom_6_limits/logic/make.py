"""Make: a squeezed joint makes more minus quanta out of itself, so it shrinks.

While the pull across a joint beats its push, every box of the joint makes whole quanta
out of its own: what the joint gives in a step (`given`) times how hard the volumes'
fluid presses on the box's row (`pressed`). What is made is booked in `cell_made`;
nothing is ever unmade, and the volumes lose nothing. Counted a turn of a swirl, not a
step: boxes are not space.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_4_crossing.definitions.facing import faced
from physics.engine.axiom_4_crossing.definitions.passage import row_of, stride_of
from physics.engine.axiom_5_packing.definitions.pull_push import given, pressed


@wp.kernel
def _make(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
          tile_slots: wp.array(dtype=wp.int32), joint_ends: wp.array(dtype=wp.vec3i),
          joint_part: wp.array2d(dtype=wp.vec4), across: wp.int32,
          seed: wp.uint32, step: wp.uint32, a_turn: wp.float64,
          own_squeeze: wp.array(dtype=wp.vec4d),
          cell_quanta: wp.array(dtype=wp.int32), cell_made: wp.array(dtype=wp.int32),
          box_quanta: wp.array2d(dtype=wp.int32)):
    slot, first, along, third = wp.tid()
    gives = given(own_squeeze[slot], a_turn)
    if gives <= wp.float64(0.0):
        return  # not squeezed: the joint's energy stays as it is
    joint, axis = own_cell[slot], wp.int32(own_squeeze[slot][3])
    stride = stride_of(across, axis)
    row = row_of(first, third, across, axis)  # the row's first box, facing the volume before
    box = row + along * stride
    share = gives * pressed(box_quanta, cell_quanta, tile_home, tile_slots, joint_ends[joint],
                            faced(joint_part, joint, first, third, across, axis))
    chance = draw_uniform(seed, step, wp.uint32(8),
                          wp.uint64(joint) * wp.uint64(box_quanta.shape[1]) + wp.uint64(box))
    made = wp.int32(wp.floor(wp.float64(-box_quanta[slot, box]) * share + chance))
    box_quanta[slot, box] -= made
    wp.atomic_add(cell_quanta, joint, -made)
    wp.atomic_add(cell_made, joint, made)


def make(store: Store) -> None:
    """One step of making in every own joint that `squeeze` found squeezed."""
    across = store.across
    wp.launch(_make, dim=(store.count, across, across, across), device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots,
                      store.joint_ends, store.joint_part, across, store.seed,
                      store.step, wp.float64(store.turn_per_step / (2.0 * wp.pi)),
                      store.own_squeeze, store.cell_quanta, store.cell_made, store.box_quanta])
