"""COARSE and REUSED blocks replace their FULL rows: the rows' memory is given back, every quantum kept.

The wrapped-cell field (64 filled blocks of a 6^3 grid) loaded FULL, then switched with
quiet_steps 1: every filled block goes COARSE, the FULL pool shrinks to its 16-row floor
and the allocated bytes a box fall to what COARSE rows cost, while the plus and minus
totals stay exact and the field reads back the same before and after the pool moved.
A rewrite and a promote bring blocks back FULL (the pool grows again) and the next pass
frees it again, totals exact throughout.
A tiled field of 64 equal blocks goes REUSED: one shared row, both pools at their floor.
A whirl field keeps its busy blocks FULL; the rest leave and the live FULL rows move down
to the low slots, read back bit for bit.
"""
from __future__ import annotations

import numpy as np

from physics.ladder import Level, LadderSettings, load_region, memory_per_box, new_store, promote_blocks
from physics.ladder import read_region, switch_levels, total_quanta, write_blocks
from physics.ladder.tests._fields import DEVICE, cell_field


def _pool_bytes(pool) -> int:
    """Device bytes a pool holds, spare rows included."""
    return pool.capacity * pool.slot_bytes()


def test_coarse_blocks_free_their_full_rows_and_keep_every_quantum():
    quanta, phase = cell_field()
    store = new_store((6, 6, 6), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(store, quanta, phase)
    totals = total_quanta(store)
    full_before = _pool_bytes(store.full)
    allocated_before = memory_per_box(store)["allocated"]
    assert totals == (int(quanta[quanta > 0].sum()), int(-quanta[quanta < 0].sum()))

    report = switch_levels(store)
    assert report["coarse"] == 64 and report["full"] == 0
    assert store.full.capacity == 16 and _pool_bytes(store.full) < full_before / 4
    after = memory_per_box(store)
    assert after["allocated"] < allocated_before / 2
    assert store.coarse.in_use() == 64 and store.coarse.capacity <= 80
    assert total_quanta(store) == totals
    decoded_q, decoded_p = read_region(store)
    assert np.array_equal(np.sign(decoded_q), np.sign(quanta))

    block = np.array([[2, 2, 2]])
    changed_q = decoded_q[16:24, 16:24, 16:24][None].copy()
    giver, taker = np.argwhere(np.sign(changed_q[0]) == np.sign(changed_q[0, 0, 0, 0]))[:2]
    kind = np.sign(changed_q[0, 0, 0, 0])
    changed_q[(0, *giver)] -= kind
    changed_q[(0, *taker)] += kind
    assert write_blocks(store, block, changed_q, decoded_p[16:24, 16:24, 16:24][None]) == 1
    assert store.level[2, 2, 2] == Level.FULL and total_quanta(store) == totals

    promote_blocks(store, np.argwhere(store.level != Level.EMPTY))
    assert store.full.in_use() == 64 and store.coarse.in_use() == 0
    promoted_q, promoted_p = read_region(store)
    expected_q = decoded_q.copy()
    expected_q[16:24, 16:24, 16:24] = changed_q[0]
    assert np.array_equal(promoted_q, expected_q) and np.array_equal(promoted_p, decoded_p)
    assert total_quanta(store) == totals

    assert switch_levels(store)["coarse"] == 64
    assert store.full.capacity == 16 and total_quanta(store) == totals


def test_reused_blocks_hold_one_shared_row_and_no_full_rows():
    tile = np.full((8, 8, 8), -700, np.int32)
    tile[3, 4, 5] = 9
    quanta = np.tile(tile, (4, 4, 4))
    phase = np.zeros(quanta.shape, np.float32)
    store = new_store((4, 4, 4), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(store, quanta, phase)
    totals = total_quanta(store)
    assert switch_levels(store)["reused"] == 64
    assert store.full.capacity == 16 and store.full.in_use() == 0
    assert store.shared.capacity == 16 and store.shared.in_use() == 1
    assert list(store.shared_refs.items()) == [(0, 64)] and set(store.shared_key.values()) == {0}
    read_q, read_p = read_region(store)
    assert np.array_equal(read_q, quanta) and np.array_equal(read_p, phase)
    assert total_quanta(store) == totals


def test_live_full_rows_move_down_bit_for_bit_when_the_pool_shrinks():
    gx, gy, gz = np.meshgrid(np.arange(64), np.arange(64), np.arange(16), indexing="ij")
    quanta = np.full(gx.shape, 2000, dtype=np.int32)
    phase = np.arctan2(gy - 31.5, gx - 31.5).astype(np.float32)
    store = new_store((8, 8, 2), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(store, quanta, phase)
    capacity_before = store.full.capacity
    switch_levels(store)
    live = store.slot[store.level == Level.FULL]
    assert 0 < live.size < capacity_before / 2
    assert store.full.capacity < capacity_before and sorted(live.tolist()) == list(range(live.size))
    read_q, read_p = read_region(store)
    near = np.hypot(gx - 31.5, gy - 31.5) < 16
    assert np.array_equal(read_q[near], quanta[near]) and np.array_equal(read_p[near], phase[near])
    assert total_quanta(store) == (int(quanta.sum()), 0)
