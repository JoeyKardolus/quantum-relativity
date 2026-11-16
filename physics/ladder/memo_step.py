"""Memo step: run a step rule on gathered blocks, computing each distinct block-with-halo only once.

Contract: `memo_step(memo, quanta, phase, step_fn)` takes gathered blocks with a halo
(`gather_blocks(store, ids, halo)`: (n, E, E, E) arrays, E = 8 + 2 halo, the halo at
least the rule's reach per step) and a rule `step_fn(quanta, phase) -> (quanta, phase)`
mapping (m, E, E, E) Warp arrays to the blocks' next content, (m, 8, 8, 8) Warp or numpy
arrays. Blocks whose exact bytes (with halo) are in `memo.table` take the stored result;
the distinct rest go to `step_fn` once each, in one batch, and are stored. Returns host
(quanta int32, phase float32) of shape (n, 8, 8, 8), bit-identical to calling `step_fn`
on every block, provided `step_fn` treats each block on its own and is deterministic.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder.settings import BLOCK
from physics.ladder.step_memo import StepMemo


def memo_step(memo: StepMemo, quanta, phase, step_fn) -> tuple:
    """Next content of every gathered block, each distinct block-with-halo computed once."""
    host_q = quanta.numpy() if isinstance(quanta, wp.array) else np.asarray(quanta)
    host_p = phase.numpy() if isinstance(phase, wp.array) else np.asarray(phase)
    host_q, host_p = host_q.astype(np.int32, copy=False), host_p.astype(np.float32, copy=False)
    count = host_q.shape[0]
    keys = [memo.rule_key + host_q[number].tobytes() + host_p[number].tobytes() for number in range(count)]
    missing: dict = {}
    for number, key in enumerate(keys):
        if key in memo.table:
            memo.hits += 1
        elif key not in missing:
            missing[key] = number
            memo.misses += 1
        else:
            memo.hits += 1
    fresh = _run_rule(quanta, phase, host_q, host_p, list(missing.values()), step_fn)
    if len(memo.table) + len(missing) > memo.max_entries:
        memo.table.clear()
    computed = {key: (fresh[0][place], fresh[1][place]) for place, key in enumerate(missing)}
    memo.table.update(computed)
    out_q = np.empty((count, BLOCK, BLOCK, BLOCK), dtype=np.int32)
    out_p = np.empty((count, BLOCK, BLOCK, BLOCK), dtype=np.float32)
    for number, key in enumerate(keys):
        out_q[number], out_p[number] = computed[key] if key in computed else memo.table[key]
    return out_q, out_p


def _run_rule(quanta, phase, host_q: np.ndarray, host_p: np.ndarray, picks: list, step_fn) -> tuple:
    """The rule's result for the picked blocks, as host int32 and float32 arrays."""
    if not picks:
        empty = np.zeros((0, BLOCK, BLOCK, BLOCK))
        return empty.astype(np.int32), empty.astype(np.float32)
    device = quanta.device if isinstance(quanta, wp.array) else None
    batch_q = wp.array(np.ascontiguousarray(host_q[picks]), dtype=wp.int32, device=device)
    batch_p = wp.array(np.ascontiguousarray(host_p[picks]), dtype=wp.float32, device=device)
    result_q, result_p = step_fn(batch_q, batch_p)
    result_q = result_q.numpy() if isinstance(result_q, wp.array) else np.asarray(result_q)
    result_p = result_p.numpy() if isinstance(result_p, wp.array) else np.asarray(result_p)
    return result_q.astype(np.int32), result_p.astype(np.float32)
