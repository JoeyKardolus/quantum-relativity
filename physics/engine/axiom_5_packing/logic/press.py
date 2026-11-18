"""Press: the squeeze on a joint turns its facing fluid on, so the fluid is pushed inward.

The same squeeze that makes the joint's quanta acts on its fluid: each step the phase of
the joint's boxes that face the volumes is turned on by what the joint gives in that step
(`given`), as a share of a whole turn, times how hard the volumes' fluid presses on that
row (`pressed`). Fluid flows from the phase that is ahead, so it is pushed inward from the
two sides, unevenly where the volumes' fluid is uneven. No press once the push answers.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.grid import cell_beside
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_4_crossing.definitions.passage import row_of, stride_of
from physics.engine.axiom_5_packing.definitions.pull_push import given, pressed


@wp.func
def _turned(phase: wp.float32, turn: float) -> wp.float32:
    """A phase turned on, kept in (-pi, pi]."""
    angle = phase + turn
    return angle - float(wp.tau) * wp.floor((angle + float(wp.pi)) / float(wp.tau))


@wp.kernel
def _press(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
           tile_slots: wp.array(dtype=wp.int32), grid: wp.vec3i, across: wp.int32,
           a_turn: wp.float64, own_squeeze: wp.array(dtype=wp.vec4d),
           cell_quanta: wp.array(dtype=wp.int32),
           box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, first, third = wp.tid()
    gives = given(own_squeeze[slot], a_turn)
    if gives <= wp.float64(0.0):
        return  # not squeezed: nothing presses
    axis, joint = wp.int32(own_squeeze[slot][3]), own_cell[slot]
    row = row_of(first, third, across, axis)        # the joint's box that faces the volume before
    last = row + (across - 1) * stride_of(across, axis)  # and the one that faces the one after
    turn = float(wp.float64(wp.tau) * gives * pressed(
        box_quanta, cell_quanta, tile_home, tile_slots, cell_beside(grid, joint, axis, -1),
        cell_beside(grid, joint, axis, 1), row, last))
    box_phase[slot, row] = _turned(box_phase[slot, row], turn)
    box_phase[slot, last] = _turned(box_phase[slot, last], turn)


def press(store: Store) -> None:
    """One step of the squeeze's press on the fluid of every own joint `squeeze` found squeezed."""
    wp.launch(_press, dim=(store.count, store.across, store.across),
              device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots,
                      wp.vec3i(*store.grid), store.across,
                      wp.float64(store.turn_per_step / (2.0 * wp.pi)), store.own_squeeze,
                      store.cell_quanta, store.box_quanta, store.box_phase])
