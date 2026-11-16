"""Gather blocks: dense quanta and phase of chosen blocks plus a halo, read from whatever level holds them.

Contract: `gather_blocks(store, block_ids, halo=0)` takes an (n, 3) array of block
coordinates and returns (quanta, phase), Warp arrays on the store's device of shape
(n, E, E, E), E = 8 + 2 halo, indexed [n, x, y, z] from the block's lowest box minus
`halo`. FULL and REUSED blocks read exactly; COARSE blocks decode with their exact
totals (`_kernels.py`); EMPTY blocks and boxes beyond the grid read 0 quanta, phase 0.
This is how the core engine reads a block and its neighbours for one step.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder._kernels import gather_kernel
from physics.ladder.settings import BLOCK
from physics.ladder.store import LadderStore


def gather_blocks(store: LadderStore, block_ids, halo: int = 0):
    """Quanta and phase of each listed block and `halo` boxes round it, as (n, E, E, E) device arrays."""
    wanted = np.ascontiguousarray(np.asarray(block_ids, dtype=np.int32).reshape(-1, 3))
    edge = BLOCK + 2 * int(halo)
    count = wanted.shape[0]
    quanta = wp.zeros((count, edge, edge, edge), dtype=wp.int32, device=store.device)
    phase = wp.zeros((count, edge, edge, edge), dtype=wp.float32, device=store.device)
    if count == 0:
        return quanta, phase
    full, shared, coarse = store.full.arrays, store.shared.arrays, store.coarse.arrays
    wp.launch(gather_kernel, dim=(count, edge, edge, edge), device=store.device,
              inputs=[store.level_dev, store.slot_dev, full["quanta"], full["phase"], shared["quanta"],
                      shared["phase"], coarse["sign"], coarse["mag"], coarse["phase"], coarse["head"],
                      coarse["phead"], wp.array(wanted, dtype=wp.int32, device=store.device), int(halo)],
              outputs=[quanta, phase])
    return quanta, phase
