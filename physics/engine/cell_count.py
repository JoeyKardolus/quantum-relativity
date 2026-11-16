"""Cell count: every cell's quanta, boxes and summed pull, counted from its boxes on the GPU.

Contract: `cell_count(state)` adds up, per universe and cell id, the quanta ignoring sign
and the boxes of every box the cell owns, and the pull over its boxes (summed as whole
millionths, int64, so the sum is the same in any order and a universe counts the same
alone as in a batch), writes the first two to state.cell_quanta and state.cell_boxes
(int64 numpy, (universes, capacity)) and returns the mean pull per cell (float64 numpy,
0 for a cell without boxes). Empty boxes count for no cell.
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
    ti, tj, tk = wp.tid()
    cell = owner[ti, tj, tk]
    if cell < 0:
        return
    home = ti // side
    wp.atomic_add(amounts, home, cell, wp.int64(wp.abs(quanta[ti, tj, tk])))
    wp.atomic_add(boxes, home, cell, wp.int64(1))
    wp.atomic_add(pulls, home, cell, wp.int64(wp.round(float(pull[ti, tj, tk]) * PULL_UNIT)))


def cell_count(state: State) -> np.ndarray:
    """Count every cell's quanta and boxes into the state; return each cell's mean pull."""
    shape = (state.universes, state.capacity)
    device = state.quanta.device
    amounts = wp.zeros(shape, dtype=wp.int64, device=device)
    boxes = wp.zeros(shape, dtype=wp.int64, device=device)
    pulls = wp.zeros(shape, dtype=wp.int64, device=device)
    wp.launch(_count, dim=state.shape, device=device,
              inputs=[state.quanta, state.owner, state.pull, state.side, amounts, boxes, pulls])
    state.cell_quanta = amounts.numpy()
    state.cell_boxes = boxes.numpy()
    return pulls.numpy() / PULL_UNIT / np.maximum(state.cell_boxes, 1)
