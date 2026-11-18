"""Press: the two volumes beside a joint, pulling on each other across it, press on its fluid.

The press acts on the joint's boxes that face the volumes: each step it turns their phase
on, ahead of the boxes behind them, and fluid flows from the phase that is ahead, so the
joint's fluid is pushed inward from its two sides. How hard follows the plus quanta that
face each other across that row. Only phases change: the joint's quanta stay its own.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import TURN
from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import slot_at

# The one setting, chosen before the first run and not moved: where two calm volumes face
# each other the press turns a phase by a tenth of the fluid's own turn, small beside the
# fluid's own motion (the owner: even the least press makes the joint's swirl irregular).
PRESS = TURN / 10.0


@wp.func
def _pressed(phase: wp.float32, pull: float) -> wp.float32:
    """A phase turned on by the press, kept in (-pi, pi]."""
    angle = phase + float(PRESS) * pull
    return angle - float(wp.tau) * wp.floor((angle + float(wp.pi)) / float(wp.tau))


@wp.kernel
def _press(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
           tile_slots: wp.array(dtype=wp.int32), across: wp.int32, cells: wp.int32,
           box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, first, third = wp.tid()
    joint = own_cell[slot]
    if joint % 2 == 0 or joint + 1 >= cells:
        return  # only a cell at an odd place has a volume on both sides
    left = slot_at(tile_home, tile_slots, joint - 1)
    right = slot_at(tile_home, tile_slots, joint + 1)
    if left < 0 or right < 0:
        return
    row = first * across * across + third  # the joint's box that faces the left volume
    last = row + (across - 1) * across     # and the one that faces the right volume
    pull = float(box_quanta[left, last] + box_quanta[right, row]) / float(2 * QUANTA_A_BOX)
    box_phase[slot, row] = _pressed(box_phase[slot, row], pull)
    box_phase[slot, last] = _pressed(box_phase[slot, last], pull)


def press(store) -> None:
    """One step of the volumes' press on every own joint that lies between two own volumes."""
    wp.launch(_press, dim=(store.count, store.across, store.across),
              device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.across,
                      store.cell_quanta.shape[0], store.box_quanta, store.box_phase])
