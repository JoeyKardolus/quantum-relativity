"""Flow inside: one step of the fluid inside every cell that holds its own boxes.

A cell's boxes are three ring whirls (`ring_boxes.py`): round each core and along each
ring the rows of boxes close on themselves; from the core outward they do not. A launch
takes every other pair of neighbours one way through the block, so no box is in two pairs
of it; round the core the even pairs go in two halves either side of the odd (`SWEEPS`). Where one ring's last layer is followed by the next
ring's first, the two are neighbours only in the jet through the cell's middle (the first
and the last box round the core): there the three rings meet and share the cell's energy.
Only the own set is visited. A joint's minus quanta move by the same law, the sign kept.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import (
    TURN, pair_rotation, phase_of, wave_of)
from physics.engine.axiom_1_energy.definitions.whole_move import odd_part, whole_move
from physics.engine.axiom_2_swirls.definitions.standing_mode import SWEEPS

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _flow(apart: wp.vec3i, parity: wp.int32, turn: wp.float64, across: wp.int32, layers: wp.int32,
          box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, pair, first, second = wp.tid()
    at = pair * 2 + parity  # parity 0 pairs boxes 0-1, 2-3, 4-5; parity 1 pairs 1-2, 3-4, 5-0
    if layers > 0 and (at + 1) % layers == 0 and first != 0 and first != across - 1:
        return  # from one ring to the next: only in the jet through the middle
    rest = first * apart[1] + second * apart[2]
    low, high = at * apart[0] + rest, (at + 1) % across * apart[0] + rest
    sign = wp.sign(box_quanta[slot, low] + box_quanta[slot, high])  # a joint's boxes are minus
    low_has = wp.abs(box_quanta[slot, low])
    high_has = wp.abs(box_quanta[slot, high])
    low_wave, high_wave = pair_rotation(wave_of(low_has, box_phase[slot, low]),
                                        wave_of(high_has, box_phase[slot, high]), turn)
    flow = wp.dot(high_wave, high_wave) - wp.float64(high_has)  # what the high box gains
    moved = whole_move(flow, odd_part(box_phase[slot, low], across), low_has, high_has)
    box_quanta[slot, low] = sign * (low_has - moved)
    box_quanta[slot, high] = sign * (high_has + moved)
    box_phase[slot, low] = phase_of(low_wave, box_phase[slot, low])
    box_phase[slot, high] = phase_of(high_wave, box_phase[slot, high])


def flow_inside(store, before_pass=None) -> None:
    """One step of the fluid inside each own cell: the pair rotation, then the whole move.

    `before_pass(axis, parity, turn)`, when given, is called before each launch, for a watcher.
    Way 0 goes from layer to layer (and ring to ring), 1 round the core, 2 along the ring.
    """
    across = store.across
    strides = (across * across, across, 1)
    for axis in range(3):
        apart = wp.vec3i(strides[axis], strides[(axis + 1) % 3], strides[(axis + 2) % 3])
        for parity, turn in (SWEEPS if axis == 1 else ((0, TURN), (1, TURN))):
            if before_pass:
                before_pass(axis, parity, turn)
            wp.launch(_flow, dim=(store.count, across // 2, across, across),
                      inputs=[apart, parity, wp.float64(turn), across, across // 3 if axis == 0 else 0,
                              store.box_quanta, store.box_phase],
                      device=store.box_quanta.device)
