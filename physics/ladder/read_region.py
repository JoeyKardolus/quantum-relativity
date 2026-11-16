"""Read region: the dense field of a range of blocks, decoded from whatever level each sits at.

Contract: `read_region(store, low_block=(0, 0, 0), high_block=None)` returns host
(quanta int32, phase float32) arrays of shape 8 x (high - low) per axis covering blocks
low <= b < high (the whole grid when high is None). EMPTY blocks read 0 quanta, phase 0;
COARSE blocks read their decoded values (exact totals, bounded error); FULL and REUSED
read exactly what was stored. For tests, the viewer and checks; the core uses
`gather_blocks`.
"""
from __future__ import annotations

import numpy as np

from physics.ladder.gather_blocks import gather_blocks
from physics.ladder.settings import BLOCK
from physics.ladder.store import LadderStore


def read_region(store: LadderStore, low_block=(0, 0, 0), high_block=None) -> tuple:
    """Dense host (quanta, phase) of blocks low <= b < high."""
    low = np.asarray(low_block, dtype=np.int64)
    high = np.asarray(store.blocks if high_block is None else high_block, dtype=np.int64)
    counts = tuple(int(size) for size in high - low)
    coords = np.argwhere(np.ones(counts, dtype=bool)) + low
    quanta, phase = gather_blocks(store, coords, halo=0)
    return _as_dense(quanta.numpy(), counts), _as_dense(phase.numpy(), counts)


def _as_dense(rows: np.ndarray, counts: tuple) -> np.ndarray:
    """(blocks, 8, 8, 8) rows in block-coordinate order back into one dense field."""
    shaped = rows.reshape(counts[0], counts[1], counts[2], BLOCK, BLOCK, BLOCK)
    return shaped.transpose(0, 3, 1, 4, 2, 5).reshape(counts[0] * BLOCK, counts[1] * BLOCK, counts[2] * BLOCK)
