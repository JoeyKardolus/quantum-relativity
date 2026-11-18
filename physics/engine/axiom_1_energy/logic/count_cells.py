"""Count cells: how many quanta each cell holds, counted from its own boxes.

Contract: `count_cells(state)` writes `cell_quanta` of every id handed out so far: the
sum of the cell's 216 boxes ignoring sign, as a whole int64, one thread per cell, so the
count is exact. An id that is no cell holds 0 in every box and so counts 0. It is the
only writer of `cell_quanta` and it writes nothing else. Reads `quanta`.
No thread reads a box of another cell.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_1_energy.definitions.cell_sum import cell_sum

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _count(quanta: wp.array4d(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64)):
    cell = wp.tid()
    cell_quanta[cell] = cell_sum(quanta, cell)


def count_cells(state) -> None:
    """Write every cell's quanta as the exact sum of its boxes."""
    wp.launch(_count, dim=state.count, inputs=[state.quanta, state.cell_quanta], device=state.device)
