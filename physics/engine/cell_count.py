"""Cell count: every cell's quanta, boxes and summed pull, counted from its boxes on the GPU.

Contract: `cell_count(state)` adds up, per universe and cell id, the quanta ignoring sign
and the boxes of every box the cell owns, and the pull over its boxes (summed as whole
millionths, int64, so the sum is the same in any order and a universe counts the same
alone as in a batch), writes the first two to state.cell_quanta and state.cell_boxes
(int64 numpy, (universes, capacity)) and returns the mean pull per cell (float64 numpy,
0 for a cell without boxes). Empty boxes count for no cell. One GPU thread walks each
line of boxes and adds a run of one cell's boxes at once (whole numbers, so the sums are
the same as box by box, with a tenth of the atomic adds).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.state import State

PULL_UNIT = 1.0e6


@wp.kernel
def _count(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32),
           side: int, amounts: wp.array2d(dtype=wp.int64), boxes: wp.array2d(dtype=wp.int64),
           pulls: wp.array2d(dtype=wp.int64)):
    ti, tj = wp.tid()
    home = ti // side
    run = int(-1)
    amount = wp.int64(0)
    held = wp.int64(0)
    pulled = wp.int64(0)
    for tk in range(side):
        cell = owner[ti, tj, tk]
        if cell != run:
            if run >= 0:
                wp.atomic_add(amounts, home, run, amount)
                wp.atomic_add(boxes, home, run, held)
                wp.atomic_add(pulls, home, run, pulled)
            run = cell
            amount = wp.int64(0)
            held = wp.int64(0)
            pulled = wp.int64(0)
        if cell >= 0:
            amount += wp.int64(wp.abs(quanta[ti, tj, tk]))
            held += wp.int64(1)
            pulled += wp.int64(wp.round(float(pull[ti, tj, tk]) * PULL_UNIT))
    if run >= 0:
        wp.atomic_add(amounts, home, run, amount)
        wp.atomic_add(boxes, home, run, held)
        wp.atomic_add(pulls, home, run, pulled)


def cell_count(state: State) -> np.ndarray:
    """Count every cell's quanta and boxes into the state; return each cell's mean pull."""
    shape = (state.universes, state.capacity)
    device = state.quanta.device
    amounts = wp.zeros(shape, dtype=wp.int64, device=device)
    boxes = wp.zeros(shape, dtype=wp.int64, device=device)
    pulls = wp.zeros(shape, dtype=wp.int64, device=device)
    wp.launch(_count, dim=state.shape[:2], device=device,
              inputs=[state.quanta, state.owner, state.pull, state.side, amounts, boxes, pulls])
    state.cell_quanta = amounts.numpy()
    state.cell_boxes = boxes.numpy()
    return pulls.numpy() / PULL_UNIT / np.maximum(state.cell_boxes, 1)
