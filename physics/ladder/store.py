"""Ladder store: the sparse block table and the slot pools behind it.

Contract: `LadderStore` is built by `new_store`. The grid is `blocks` = (nbx, nby, nbz)
blocks of 8^3 Planck boxes. The block table is a dense index, one int32 level and one
int32 slot per block (host numpy, authoritative; mirrored on the device for kernels by
`sync()`): an EMPTY block owns no row anywhere, so it costs 8 bytes of index, 1/64 byte a
box. A FULL block's slot is a row of `full` (quanta int32 and phase float32, [x, y, z]);
a COARSE block's a row of `coarse`; a REUSED block's a row of `shared`, held by every
block whose exact content (quanta and phase bytes) equals it: `shared_key` maps those
bytes to the slot, `shared_refs` counts holders. `quiet_run` counts each block's quiet
measurements in a row. Only the ladder's own functions change a store.

Why a block table and not NanoVDB (warp.Volume): a NanoVDB grid holds one value type per
voxel and its tree is fixed between rebuilds, while ladder blocks change representation
every step (full int32 + float32, packed 2/4-bit codes, a pointer to a shared copy). A
Volume's 8^3 leaf is the same size as a block, so a FULL-only region could later sit in an
index grid with the same slots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from physics.ladder._pool import _Pool
from physics.ladder.settings import LadderSettings


@dataclass
class LadderStore:
    """Block table, pools and shared-copy keys of one ladder grid."""

    settings: LadderSettings
    device: object
    blocks: tuple
    level: np.ndarray
    slot: np.ndarray
    quiet_run: np.ndarray
    full: _Pool
    coarse: _Pool
    shared: _Pool
    shared_key: dict = field(default_factory=dict)
    shared_bytes: dict = field(default_factory=dict)
    shared_refs: dict = field(default_factory=dict)
    level_dev: wp.array = None
    slot_dev: wp.array = None

    def sync(self) -> None:
        """Copy the block table to the device for the kernels."""
        self.level_dev = wp.array(self.level, dtype=wp.int32, device=self.device)
        self.slot_dev = wp.array(self.slot, dtype=wp.int32, device=self.device)
