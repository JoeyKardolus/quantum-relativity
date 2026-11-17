"""Pour: the quanta a cell gave and took this step leave and enter its own boxes, exactly; arrivals in a volume turn to heat.

Contract: `pour(state)` applies state.loss and state.gain (cross.py, squeeze.py) to each
living cell's block and to its cell_quanta. Energy moves between cells as numbers; no
box moves (axiom 3: boxes are not space). First the loss leaves: each box gives in
proportion to what it holds above 1 quantum, in whole quanta, box i giving floor(loss *
cumulative share up to i) minus what the boxes before it gave, so the boxes together
give exactly the loss and none drops below 1. Then the gain enters in the same way in
proportion to what each box holds. One thread works one cell in a fixed box order.
Friction (axiom 4): in an energy volume the m quanta arriving in a box holding n come
without their direction, at a random phase (a hash of the box, the state's key and the
step), so the box's phase becomes arg(sqrt(n) e^{i phase} + sqrt(m) e^{i random}):
directed flow turned to jiggling, heat, and no quantum lost. What a joint made is its
own energy and enters at the box's phase. Signs never change here. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.state import BLOCK, BOXES, State

wp.set_module_options({"fuse_fp": False})

PI = 3.14159265358979


@wp.kernel
def _pour(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64), gain: wp.array(dtype=wp.int64),
          loss: wp.array(dtype=wp.int64), quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float32),
          across: int, seed: int):
    cell = wp.tid()
    sort = kind[cell]
    if sort == 0:
        return
    leaving = loss[cell]
    arriving = gain[cell]
    if leaving == wp.int64(0) and arriving == wp.int64(0):
        return
    held = cell_quanta[cell]
    if leaving > wp.int64(0):
        spare = wp.float64(held - wp.int64(BOXES))
        given = wp.int64(0)
        run = wp.int64(0)
        for one in range(across):
            for two in range(across):
                for three in range(across):
                    amount = wp.abs(quanta[cell, one, two, three])
                    run += wp.int64(amount - 1)
                    aim = wp.min(wp.int64(wp.floor(wp.float64(leaving) * (wp.float64(run) / spare))), leaving)
                    part = int(wp.clamp(aim - given, wp.int64(0), wp.int64(amount - 1)))
                    given += wp.int64(part)
                    quanta[cell, one, two, three] = sort * (amount - part)
        if given < leaving:
            for one in range(across):
                for two in range(across):
                    for three in range(across):
                        amount = wp.abs(quanta[cell, one, two, three])
                        part = int(wp.min(leaving - given, wp.int64(amount - 1)))
                        given += wp.int64(part)
                        quanta[cell, one, two, three] = sort * (amount - part)
    if arriving > wp.int64(0):
        base = wp.float64(held - leaving)
        taken = wp.int64(0)
        run = wp.int64(0)
        rng = wp.rand_init(seed, cell)
        for one in range(across):
            for two in range(across):
                for three in range(across):
                    amount = wp.abs(quanta[cell, one, two, three])
                    run += wp.int64(amount)
                    aim = wp.min(wp.int64(wp.floor(wp.float64(arriving) * (wp.float64(run) / base))), arriving)
                    if one == across - 1 and two == across - 1 and three == across - 1:
                        aim = arriving
                    part = int(wp.max(aim - taken, wp.int64(0)))
                    taken += wp.int64(part)
                    quanta[cell, one, two, three] = sort * (amount + part)
                    turn = 2.0 * PI * wp.randf(rng)
                    if part > 0 and sort == 1:
                        angle = phase[cell, one, two, three]
                        old = wp.sqrt(float(amount))
                        new = wp.sqrt(float(part))
                        phase[cell, one, two, three] = wp.atan2(old * wp.sin(angle) + new * wp.sin(turn),
                                                                old * wp.cos(angle) + new * wp.cos(turn))
    cell_quanta[cell] = held - leaving + arriving


def pour(state: State) -> None:
    """Move this step's lost and gained quanta out of and into each cell's own boxes."""
    seed = (state.step * 64 + 62) ^ state.key
    wp.launch(_pour, dim=state.count, device=state.device,
              inputs=[state.kind, state.cell_quanta, state.gain, state.loss, state.quanta, state.phase, BLOCK,
                      int(seed & 0x7FFFFFFF)])
