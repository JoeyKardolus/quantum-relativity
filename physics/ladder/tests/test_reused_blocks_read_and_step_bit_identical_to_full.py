"""REUSED: identical quiet blocks share one exact copy and one step result, bit-identical to FULL.

A quiet field repeating every 8 boxes (one wrapped cell per block, a ramp of magnitudes, phase at rest) in a
6 x 6 x 4 grid: every block's content is the same, so they all become REUSED holding one
shared copy, and the field reads back bit for bit. A step rule run through `memo_step`
on the blocks gathered with a halo of 1 matches the rule run on every block on its own
bit for bit; the rule runs once per distinct block-with-halo, and blocks at the grid's
faces (whose halo reads the zero beyond it) do not share with inner ones.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder import Level, LadderSettings, StepMemo, gather_blocks, load_region, memo_step, new_store
from physics.ladder import read_region, switch_levels, total_quanta
from physics.ladder.tests._fields import DEVICE


def _rule(quanta, phase):
    """A test step: each box gives an eighth of its quanta (floored) shared over its faces; phase turns by quanta."""
    values = (quanta.numpy() if isinstance(quanta, wp.array) else quanta).astype(np.int64)
    turns = phase.numpy() if isinstance(phase, wp.array) else phase
    give = values // 8
    kept = values - give
    for axis in (1, 2, 3):
        for shift in (1, -1):
            kept = kept + np.roll(give, shift, axis=axis) // 6
    inner = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
    return kept[inner].astype(np.int32), (turns + np.float32(1e-4) * values.astype(np.float32))[inner]


def test_reused_blocks_read_and_step_bit_identical_to_full():
    gx, gy, gz = np.meshgrid(np.arange(48), np.arange(48), np.arange(32), indexing="ij")
    volume = ((gx % 8) >= 1) & ((gx % 8) <= 5) & ((gy % 8) >= 1) & ((gy % 8) <= 5) & ((gz % 8) >= 1) & ((gz % 8) <= 5)
    magnitude = 1500 + (gx % 8) + 2 * (gy % 8) + 3 * (gz % 8)
    quanta = np.where(volume, magnitude, -magnitude).astype(np.int32)
    phase = np.full(gx.shape, 0.2, dtype=np.float32)
    store = new_store((6, 6, 4), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(store, quanta, phase)
    total = total_quanta(store)
    report = switch_levels(store)
    assert report["reused"] == 6 * 6 * 4 and store.shared.in_use() == 1
    read_q, read_p = read_region(store)
    assert np.array_equal(read_q, quanta) and np.array_equal(read_p, phase)
    assert total_quanta(store) == total

    coords = np.argwhere(store.level == Level.REUSED)
    halo_q, halo_p = gather_blocks(store, coords, halo=1)
    memo = StepMemo(rule_key=b"test-rule-1")
    memo_q, memo_p = memo_step(memo, halo_q, halo_p, _rule)
    direct = [_rule(halo_q.numpy()[[number]], halo_p.numpy()[[number]]) for number in range(len(coords))]
    assert np.array_equal(memo_q, np.concatenate([part[0] for part in direct]))
    assert np.array_equal(memo_p.view(np.uint32), np.concatenate([part[1] for part in direct]).view(np.uint32))
    distinct = {halo_q.numpy()[number].tobytes() + halo_p.numpy()[number].tobytes() for number in range(len(coords))}
    assert memo.misses == len(distinct) == 3 ** 3
    assert memo.hits == len(coords) - memo.misses
    again_q, _ = memo_step(memo, halo_q, halo_p, _rule)
    assert np.array_equal(again_q, memo_q) and memo.misses == 3 ** 3
