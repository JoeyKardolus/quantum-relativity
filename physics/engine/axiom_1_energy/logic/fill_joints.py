"""Fill joints: every joint of the start gets the whole quanta it is told to hold.

Contract: `fill_joints(state, settings, levels)` rewrites the boxes of every living
joint of a start (before any step) and counts `cell_quanta` again. `levels` is a host
array, one number per cell id: the quanta the joint is to hold (only the joints' rows
are read). A joint holds `max(round(level), 216)` whole quanta, exactly: every box gets
1 (no box starts empty) and the rest is dealt over the boxes in proportion to what each
holds now, whole parts first, what is left over one each from box 0 up. So a joint
keeps the shape its fluid has (the start's unevenness from box to box, its swirl). All
minus.
Returns the largest change of any joint's quanta as a share of what it held before.
Writes the joints' `quanta` and `cell_quanta`, nothing else; an energy volume is not
touched. Raises ValueError after the first step: only a start is laid this way (axiom 6:
a living joint's energy rises only while it is squeezed).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, BOXES, JOINT

wp.set_module_options({"fuse_fp": False})

_JOINT = wp.constant(wp.int32(JOINT))
_BLOCK = wp.constant(BLOCK)
_BOXES = wp.constant(wp.uint64(BOXES))


@wp.kernel
def _deal(kind: wp.array(dtype=wp.int32), level: wp.array(dtype=wp.int64), held: wp.array(dtype=wp.int64),
          quanta: wp.array4d(dtype=wp.int32)):
    """One thread per cell id: a joint's boxes get 1 each and the rest of its level by what each holds, exactly."""
    cell = wp.tid()
    if kind[cell] != _JOINT or held[cell] <= wp.int64(0):
        return
    rest = level[cell] - wp.int64(_BOXES)
    left = rest
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                mine = wp.float64(-quanta[cell, across, along, up])
                share = wp.min(wp.int64(wp.floor(wp.float64(rest) * mine / wp.float64(held[cell]))), left)
                quanta[cell, across, along, up] = -wp.int32(share + wp.int64(1))
                left -= share
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                if left > wp.int64(0):
                    quanta[cell, across, along, up] = quanta[cell, across, along, up] - 1
                    left -= wp.int64(1)


def fill_joints(state, settings, levels) -> float:
    """Give every joint of the start the quanta `levels` says; return the largest change as a share."""
    if state.step != 0:
        raise ValueError(f"fill_joints lays a start; this universe is at step {state.step}")
    joints = state.host("kind") == JOINT
    before = state.host("cell_quanta").astype(np.float64)
    wanted = np.zeros(state.capacity, dtype=np.int64)
    wanted[:state.count][joints] = np.maximum(np.rint(np.asarray(levels, dtype=np.float64)[:state.count][joints]),
                                              BOXES).astype(np.int64)
    if wanted.max(initial=0) > (2 ** 31 - 1) * BOXES // 4:
        raise OverflowError("a joint would start with more quanta than its boxes can hold")
    wp.launch(_deal, dim=state.count, device=state.device,
              inputs=[state.kind, wp.array(wanted, dtype=wp.int64, device=state.device), state.cell_quanta,
                      state.quanta])
    count_cells(state)
    if not joints.any():
        return 0.0
    return float(np.max(np.abs(wanted[:state.count][joints] - before[joints]) / np.maximum(before[joints], 1.0)))
