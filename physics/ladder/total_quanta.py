"""Total quanta: the exact count of pulling (plus) and pushing (minus) whole quanta in the whole grid.

Contract: `total_quanta(store)` returns (plus, minus) as Python ints: the sum of every
positive box and of the magnitude of every negative box, read through each block's level
(COARSE decoded), summed in int64 on the device. Bible rule 4: plus + minus, the amount
ignoring sign, is what stays fixed; the ladder keeps plus and minus each exactly.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder._kernels import total_kernel
from physics.ladder.gather_blocks import gather_blocks
from physics.ladder.level import Level
from physics.ladder.settings import BLOCK
from physics.ladder.store import LadderStore


def total_quanta(store: LadderStore) -> tuple:
    """(plus, minus): whole quanta pulling and pushing, over every block."""
    coords = np.argwhere(store.level != Level.EMPTY)
    totals = wp.zeros(2, dtype=wp.int64, device=store.device)
    chunk = max(1, int(store.settings.chunk_blocks))
    for start in range(0, coords.shape[0], chunk):
        quanta, _ = gather_blocks(store, coords[start:start + chunk], halo=0)
        wp.launch(total_kernel, dim=quanta.shape, device=store.device, inputs=[quanta], outputs=[totals])
    plus, minus = totals.numpy().tolist()
    return int(plus), int(minus)
