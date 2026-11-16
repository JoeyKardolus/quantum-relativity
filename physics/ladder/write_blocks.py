"""Write blocks: the core's new content for some blocks, stored FULL where it changed.

Contract: `write_blocks(store, block_ids, quanta, phase)` takes an (n, 3) array of block
coordinates and the blocks' new content as (n, 8, 8, 8) device or host arrays (quanta
int32, phase float32). A block whose new content is bit-identical to what it reads now
is left untouched at its level (a quiet block at rest costs nothing and gathers no
re-coding error). A changed block is stored FULL exactly as given (a new FULL row; its
COARSE or REUSED storage is released), or EMPTY when all its quanta are 0. Its quiet run
is kept: `switch_levels` decides its level next. Returns how many blocks changed. The
quanta total is whatever the core wrote: keeping it is the core's job, the ladder only
never changes it.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder._kernels import copy_rows_kernel, differs_kernel
from physics.ladder._moves import _release, _rows
from physics.ladder.gather_blocks import gather_blocks
from physics.ladder.level import Level
from physics.ladder.settings import BLOCK
from physics.ladder.store import LadderStore


def write_blocks(store: LadderStore, block_ids, quanta, phase) -> int:
    """Store the changed blocks' new content FULL; return how many changed."""
    coords = np.asarray(block_ids, dtype=np.int64).reshape(-1, 3)
    count = coords.shape[0]
    new_q = _on_device(store, quanta, wp.int32)
    new_p = _on_device(store, phase, wp.float32)
    if new_q.shape != (count, BLOCK, BLOCK, BLOCK) or new_p.shape != new_q.shape:
        raise ValueError(f"quanta and phase must be ({count}, 8, 8, 8), got {new_q.shape}, {new_p.shape}")
    if count == 0:
        return 0
    old_q, old_p = gather_blocks(store, coords, halo=0)
    changed = wp.zeros(count, dtype=wp.int32, device=store.device)
    wp.launch(differs_kernel, dim=(count, BLOCK, BLOCK, BLOCK), device=store.device,
              inputs=[old_q, old_p, new_q, new_p], outputs=[changed])
    moved = changed.numpy() == 1
    if moved.any():
        kept_run = store.quiet_run[tuple(coords[moved].T)].copy()
        _release(store, coords[moved])
        filled = moved & np.any(new_q.numpy() != 0, axis=(1, 2, 3))
        if filled.any():
            slots = store.full.alloc(int(filled.sum()))
            picks = np.flatnonzero(filled)
            wp.launch(copy_rows_kernel, dim=(len(picks), BLOCK, BLOCK, BLOCK), device=store.device,
                      inputs=[new_q, new_p, _rows(store, picks), store.full.arrays["quanta"],
                              store.full.arrays["phase"], _rows(store, slots)])
            where = tuple(coords[filled].T)
            store.level[where] = Level.FULL
            store.slot[where] = slots
        store.quiet_run[tuple(coords[moved].T)] = kept_run
        store.sync()
    return int(moved.sum())


def _on_device(store: LadderStore, values, dtype) -> wp.array:
    """`values` as a device array of `dtype` on the store's device (no copy when it already is one)."""
    if isinstance(values, wp.array):
        if values.dtype != dtype:
            raise ValueError(f"expected {dtype}, got {values.dtype}")
        return values.to(store.device)
    return wp.array(np.ascontiguousarray(values), dtype=dtype, device=store.device)
