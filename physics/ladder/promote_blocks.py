"""Promote blocks: bring blocks to FULL detail now, ahead of the switching rules.

Contract: `promote_blocks(store, block_ids)` takes an (n, 3) array of block coordinates;
COARSE blocks decode to FULL (their decoded values, totals exact), REUSED blocks get their
own exact FULL copy, FULL and EMPTY blocks stay. Each listed block's quiet run restarts,
so it stays FULL for at least `quiet_steps` measurements. The core calls it for a region
where it knows something is about to happen (a collapse, an incoming wave).
"""
from __future__ import annotations

import numpy as np

from physics.ladder._moves import _index, _to_full
from physics.ladder.store import LadderStore


def promote_blocks(store: LadderStore, block_ids) -> None:
    """Bring the listed blocks to FULL and restart their quiet runs."""
    coords = np.asarray(block_ids, dtype=np.int64).reshape(-1, 3)
    _to_full(store, coords)
    store.quiet_run[_index(coords)] = 0
    store.sync()
