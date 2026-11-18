"""Flow inside: one step of the fluid inside every cell that holds its own boxes.

Six launches (three axes, two parities) take every face between two boxes of a block
once; no box is in two pairs of one launch. Only the own set is visited.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.pair_rotation import (
    TURN, pair_rotation, phase_of, wave_of)
from physics.engine.axiom_1_energy.definitions.whole_move import whole_move

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _flow(own_cell: wp.array(dtype=wp.int32), apart: wp.vec3i, parity: wp.int32,
          pass_id: wp.uint32, seed: wp.uint32, step: wp.uint32,
          box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, pair, first, second = wp.tid()
    # along the axis, parity 0 pairs boxes 0-1, 2-3, 4-5 and parity 1 pairs 1-2, 3-4
    low = (pair * 2 + parity) * apart[0] + first * apart[1] + second * apart[2]
    high = low + apart[0]
    low_has = box_quanta[slot, low]
    high_has = box_quanta[slot, high]
    low_wave, high_wave = pair_rotation(wave_of(low_has, box_phase[slot, low]),
                                        wave_of(high_has, box_phase[slot, high]), wp.float64(TURN))
    box = wp.uint64(own_cell[slot]) * wp.uint64(box_quanta.shape[1]) + wp.uint64(low)
    flow = wp.dot(high_wave, high_wave) - wp.float64(high_has)  # what the high box gains
    moved = whole_move(flow, draw_uniform(seed, step, pass_id, box), low_has, high_has)
    box_quanta[slot, low] = low_has - moved
    box_quanta[slot, high] = high_has + moved
    box_phase[slot, low] = phase_of(low_wave, box_phase[slot, low])
    box_phase[slot, high] = phase_of(high_wave, box_phase[slot, high])


def flow_inside(store) -> None:
    """One step of the fluid inside each own cell: the pair rotation, then the whole move."""
    across = store.across
    strides = (across * across, across, 1)  # how far apart neighbours along each axis lie
    for axis in range(3):
        # the pairs' own axis first, then the two other axes
        apart = wp.vec3i(strides[axis], strides[(axis + 1) % 3], strides[(axis + 2) % 3])
        for parity in (0, 1):
            pairs = (across - parity) // 2
            wp.launch(_flow, dim=(store.count, pairs, across, across),
                      inputs=[store.own_cell, apart, parity, axis * 2 + parity, store.seed,
                              store.step, store.box_quanta, store.box_phase],
                      device=store.box_quanta.device)
