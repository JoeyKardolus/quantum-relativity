"""Load region: put a dense field of whole quanta and phase into the ladder as FULL blocks.

Contract: `load_region(store, quanta, phase, origin_block=(0, 0, 0))` takes host arrays
of shape (X, Y, Z), each a multiple of 8: `quanta` signed whole numbers (plus pulls, an
energy volume; minus pushes, a joint; 0 no space), fitting int32, and `phase` (radians,
stored float32). The blocks it covers, from `origin_block`, are replaced: those holding
any nonzero quanta become FULL, exactly as given; all-zero blocks become EMPTY and cost
no storage. Levels below FULL come only from `switch_levels`.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder._kernels import copy_rows_kernel
from physics.ladder._moves import _release, _rows
from physics.ladder.level import Level
from physics.ladder.settings import BLOCK
from physics.ladder.store import LadderStore


def load_region(store: LadderStore, quanta, phase, origin_block=(0, 0, 0)) -> None:
    """Replace the covered blocks with this dense field, nonzero blocks FULL, all-zero ones EMPTY."""
    quanta, phase = np.asarray(quanta), np.asarray(phase)
    if quanta.ndim != 3 or quanta.shape != phase.shape or any(size % BLOCK for size in quanta.shape):
        raise ValueError(f"quanta and phase must share a 3D shape of multiples of {BLOCK}, got {quanta.shape}, {phase.shape}")
    if quanta.size and (quanta.max() > np.iinfo(np.int32).max or quanta.min() < -np.iinfo(np.int32).max):
        raise ValueError("quanta must fit int32")
    counts = tuple(size // BLOCK for size in quanta.shape)
    origin = np.asarray(origin_block, dtype=np.int64)
    if np.any(origin < 0) or np.any(origin + counts > np.asarray(store.blocks)):
        raise ValueError(f"region of {counts} blocks at {tuple(origin)} does not fit the grid {store.blocks}")
    rows_q = _as_rows(quanta.astype(np.int32), counts)
    rows_p = _as_rows(phase.astype(np.float32), counts)
    coords = np.argwhere(np.ones(counts, dtype=bool)) + origin
    _release(store, coords)
    filled = np.any(rows_q != 0, axis=(1, 2, 3))
    count = int(filled.sum())
    if count:
        slots = store.full.alloc(count)
        source_q = wp.array(np.ascontiguousarray(rows_q[filled]), dtype=wp.int32, device=store.device)
        source_p = wp.array(np.ascontiguousarray(rows_p[filled]), dtype=wp.float32, device=store.device)
        wp.launch(copy_rows_kernel, dim=(count, BLOCK, BLOCK, BLOCK), device=store.device,
                  inputs=[source_q, source_p, _rows(store, np.arange(count)), store.full.arrays["quanta"],
                          store.full.arrays["phase"], _rows(store, slots)])
        where = tuple(coords[filled].T)
        store.level[where] = Level.FULL
        store.slot[where] = slots
        store.quiet_run[where] = 0
    store.sync()


def _as_rows(values: np.ndarray, counts: tuple) -> np.ndarray:
    """A dense (X, Y, Z) field cut into (blocks, 8, 8, 8) rows in block-coordinate order."""
    shaped = values.reshape(counts[0], BLOCK, counts[1], BLOCK, counts[2], BLOCK)
    return shaped.transpose(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK, BLOCK, BLOCK)
