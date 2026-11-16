"""New store: an empty ladder grid of blocks, every block EMPTY.

Contract: `new_store(blocks, settings=None, device=None)` takes the grid size in blocks
(nbx, nby, nbz), each block 8^3 Planck boxes, and returns a `LadderStore` with every block
EMPTY (all quanta 0, no space) and empty pools on `device` (Warp's default when None).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder._pool import _Pool
from physics.ladder.settings import BLOCK, GROUP, LadderSettings
from physics.ladder.store import LadderStore

_GROUPS = (BLOCK // GROUP) ** 3
_BOXES = BLOCK ** 3


def new_store(blocks, settings: LadderSettings | None = None, device=None) -> LadderStore:
    """An all-EMPTY ladder grid of `blocks` = (nbx, nby, nbz) blocks."""
    blocks = tuple(int(count) for count in blocks)
    if len(blocks) != 3 or min(blocks) < 1:
        raise ValueError(f"blocks must be three positive counts, got {blocks}")
    device = wp.get_device(device)
    box_row = (BLOCK, BLOCK, BLOCK)
    full = _Pool({"quanta": (wp.int32, box_row), "phase": (wp.float32, box_row)}, device)
    shared = _Pool({"quanta": (wp.int32, box_row), "phase": (wp.float32, box_row)}, device)
    coarse = _Pool({"sign": (wp.uint8, (_BOXES // 4,)), "mag": (wp.uint8, (_BOXES // 2,)),
                    "phase": (wp.uint8, (_BOXES // 2,)), "head": (wp.int32, (_GROUPS * 6,)),
                    "phead": (wp.float32, (_GROUPS * 2,))}, device)
    store = LadderStore(settings=settings or LadderSettings(), device=device, blocks=blocks,
                        level=np.zeros(blocks, dtype=np.int32), slot=np.full(blocks, -1, dtype=np.int32),
                        quiet_run=np.zeros(blocks, dtype=np.int32), full=full, coarse=coarse, shared=shared)
    store.sync()
    return store
