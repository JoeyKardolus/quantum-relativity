"""Memory per box: the bytes each detail level costs a Planck box, from the store's real arrays.

Contract: `memory_per_box(store)` returns a dict:
  full, coarse          device bytes one block's row takes at that level, per box
  reused                device bytes of the shared copies in use, over all boxes held
                        REUSED (0.0 when none); reused_host the host key bytes likewise
  empty                 0.0: an EMPTY block owns no row
  index                 bytes of the block table (host level, slot, quiet run; device
                        level, slot) per box of the whole grid
  in_use                device bytes of the rows in use plus the device index, per box of
                        the whole grid; allocated the same with the pools' spare capacity
  blocks_<level>        how many blocks sit at each level
"""
from __future__ import annotations

from physics.ladder.level import Level
from physics.ladder.settings import BOXES
from physics.ladder.store import LadderStore


def memory_per_box(store: LadderStore) -> dict:
    """Bytes per Planck box at each level and overall, measured from the store's arrays."""
    grid_boxes = store.level.size * BOXES
    reused_boxes = int((store.level == Level.REUSED).sum()) * BOXES
    host_index = store.level.nbytes + store.slot.nbytes + store.quiet_run.nbytes
    device_index = store.level_dev.capacity + store.slot_dev.capacity
    pools = (store.full, store.coarse, store.shared)
    in_use = sum(pool.in_use() * pool.slot_bytes() for pool in pools) + device_index
    allocated = sum(pool.capacity * pool.slot_bytes() for pool in pools) + device_index
    result = {
        "full": store.full.slot_bytes() / BOXES,
        "coarse": store.coarse.slot_bytes() / BOXES,
        "reused": store.shared.in_use() * store.shared.slot_bytes() / reused_boxes if reused_boxes else 0.0,
        "reused_host": sum(len(key) for key in store.shared_key) / reused_boxes if reused_boxes else 0.0,
        "empty": 0.0,
        "index": (host_index + device_index) / grid_boxes,
        "in_use": in_use / grid_boxes,
        "allocated": allocated / grid_boxes,
    }
    for level in Level:
        result[f"blocks_{level.name.lower()}"] = int((store.level == level).sum())
    return result
