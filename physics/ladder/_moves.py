"""Moves between levels: every switch of a block's detail goes through these, and none changes its quanta.

`_release` frees a block's storage (to EMPTY), `_to_full` decodes COARSE or copies
REUSED into a FULL row, `_to_coarse` encodes FULL blocks and keeps those within the
error bound, `_to_shared` hands FULL blocks to a shared exact copy keyed by their bytes,
`_full_rows` copies FULL rows to the host, `_compact` gives freed rows' memory back. They
change the host block table only; the caller syncs it to the device once at the end.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder._kernels import copy_rows_kernel, decode_kernel, encode_kernel
from physics.ladder.level import Level
from physics.ladder.settings import BLOCK, GROUP
from physics.ladder.store import LadderStore

_GROUPS = (BLOCK // GROUP) ** 3


def _index(coords: np.ndarray) -> tuple:
    """Numpy fancy index of (n, 3) block coordinates."""
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    return coords[:, 0], coords[:, 1], coords[:, 2]


def _rows(store: LadderStore, slots: np.ndarray) -> wp.array:
    """Slot numbers as a device array."""
    return wp.array(np.ascontiguousarray(slots, dtype=np.int32), dtype=wp.int32, device=store.device)


def _drop_shared(store: LadderStore, slots: np.ndarray) -> None:
    """One holder fewer for each shared slot; a slot nobody holds is freed with its key."""
    for slot in np.asarray(slots).ravel().tolist():
        store.shared_refs[slot] -= 1
        if store.shared_refs[slot] == 0:
            del store.shared_refs[slot]
            del store.shared_key[store.shared_bytes.pop(slot)]
            store.shared.release([slot])


def _release(store: LadderStore, coords) -> None:
    """Free the storage of these blocks and mark them EMPTY."""
    where = _index(coords)
    levels, slots = store.level[where], store.slot[where]
    store.full.release(slots[levels == Level.FULL])
    store.coarse.release(slots[levels == Level.COARSE])
    _drop_shared(store, slots[levels == Level.REUSED])
    store.level[where] = Level.EMPTY
    store.slot[where] = -1


def _to_full(store: LadderStore, coords) -> None:
    """Bring COARSE and REUSED blocks to FULL; FULL and EMPTY blocks are left as they are."""
    where = _index(coords)
    levels, slots = store.level[where], store.slot[where]
    full = store.full
    for code in (Level.COARSE, Level.REUSED):
        pick = levels == code
        count = int(pick.sum())
        if count == 0:
            continue
        old, new = slots[pick], full.alloc(count)
        if code == Level.COARSE:
            arrays = store.coarse.arrays
            wp.launch(decode_kernel, dim=(count, BLOCK, BLOCK, BLOCK), device=store.device,
                      inputs=[arrays["sign"], arrays["mag"], arrays["phase"], arrays["head"], arrays["phead"],
                              _rows(store, old), full.arrays["quanta"], full.arrays["phase"], _rows(store, new)])
            store.coarse.release(old)
        else:
            shared = store.shared.arrays
            wp.launch(copy_rows_kernel, dim=(count, BLOCK, BLOCK, BLOCK), device=store.device,
                      inputs=[shared["quanta"], shared["phase"], _rows(store, old), full.arrays["quanta"],
                              full.arrays["phase"], _rows(store, new)])
            _drop_shared(store, old)
        sub = tuple(axis[pick] for axis in where)
        store.level[sub] = Level.FULL
        store.slot[sub] = new


def _full_rows(store: LadderStore, slots: np.ndarray) -> tuple:
    """Host copies (quanta, phase), shape (n, 8, 8, 8), of FULL rows `slots`."""
    count = len(slots)
    quanta = wp.zeros((count, BLOCK, BLOCK, BLOCK), dtype=wp.int32, device=store.device)
    phase = wp.zeros((count, BLOCK, BLOCK, BLOCK), dtype=wp.float32, device=store.device)
    if count:
        wp.launch(copy_rows_kernel, dim=(count, BLOCK, BLOCK, BLOCK), device=store.device,
                  inputs=[store.full.arrays["quanta"], store.full.arrays["phase"], _rows(store, slots), quanta,
                          phase, _rows(store, np.arange(count))])
    return quanta.numpy(), phase.numpy()


def _to_coarse(store: LadderStore, coords) -> np.ndarray:
    """Encode FULL blocks as COARSE; keep those whose every group is within the error bound. Returns that mask."""
    where = _index(coords)
    count = len(where[0])
    if count == 0:
        return np.zeros(0, dtype=bool)
    old, new = store.slot[where], store.coarse.alloc(count)
    within = wp.zeros((count, _GROUPS), dtype=wp.int32, device=store.device)
    arrays, settings = store.coarse.arrays, store.settings
    wp.launch(encode_kernel, dim=(count, _GROUPS), device=store.device,
              inputs=[store.full.arrays["quanta"], store.full.arrays["phase"], _rows(store, old), _rows(store, new),
                      float(settings.max_rel_error), float(settings.max_phase_error)],
              outputs=[arrays["sign"], arrays["mag"], arrays["phase"], arrays["head"], arrays["phead"], within])
    kept = within.numpy().min(axis=1) == 1
    store.coarse.release(new[~kept])
    store.full.release(old[kept])
    sub = tuple(axis[kept] for axis in where)
    store.level[sub] = Level.COARSE
    store.slot[sub] = new[kept]
    return kept


def _to_shared(store: LadderStore, coords, keys: list) -> None:
    """Hand FULL blocks to the shared exact copy their content key names, making the copy when it is new."""
    where = _index(coords)
    for number, key in enumerate(keys):
        block = tuple(int(axis[number]) for axis in where)
        old = int(store.slot[block])
        if key in store.shared_key:
            slot = store.shared_key[key]
        else:
            slot = int(store.shared.alloc(1)[0])
            shared = store.shared.arrays
            wp.launch(copy_rows_kernel, dim=(1, BLOCK, BLOCK, BLOCK), device=store.device,
                      inputs=[store.full.arrays["quanta"], store.full.arrays["phase"], _rows(store, [old]),
                              shared["quanta"], shared["phase"], _rows(store, [slot])])
            store.shared_key[key] = slot
            store.shared_bytes[slot] = key
            store.shared_refs[slot] = 0
        store.shared_refs[slot] += 1
        store.full.release([old])
        store.level[block] = Level.REUSED
        store.slot[block] = slot


def _compact(store: LadderStore) -> None:
    """Shrink every pool to its live rows (`_Pool.shrink`) and renumber the block table and shared keys."""
    for code, pool in ((Level.FULL, store.full), (Level.COARSE, store.coarse)):
        held = store.level == code
        remap = pool.shrink(store.slot[held])
        if remap is not None:
            store.slot[held] = remap[store.slot[held]]
    held = store.level == Level.REUSED
    remap = store.shared.shrink(np.fromiter(store.shared_refs, dtype=np.int32, count=len(store.shared_refs)))
    if remap is None:
        return
    store.slot[held] = remap[store.slot[held]]
    store.shared_refs = {int(remap[slot]): refs for slot, refs in store.shared_refs.items()}
    store.shared_bytes = {int(remap[slot]): key for slot, key in store.shared_bytes.items()}
    store.shared_key = {key: int(remap[slot]) for key, slot in store.shared_key.items()}
