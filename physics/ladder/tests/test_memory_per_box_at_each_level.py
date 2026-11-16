"""Memory per Planck box at each level, read from the store's real arrays.

FULL 8 bytes a box (int32 quanta, float32 phase); COARSE 1.75 (2-bit sign, 4-bit
magnitude and phase offsets, 32 bytes of shared scales per 64-box group); REUSED one
shared 4 KiB copy over every block holding it; EMPTY no row, only its 8-byte index entry
(about 0.04 bytes a box with the host's quiet-run counter and the device mirror).
"""
from __future__ import annotations

import numpy as np

from physics.ladder import LadderSettings, load_region, memory_per_box, new_store, switch_levels
from physics.ladder.tests._fields import DEVICE, cell_field


def test_memory_per_box_at_each_level():
    empty = new_store((8, 8, 8), device=DEVICE)
    blank = memory_per_box(empty)
    assert blank["full"] == 8.0 and blank["coarse"] == 1.75 and blank["empty"] == 0.0
    assert blank["in_use"] == 2 * 4 / 512 and blank["index"] < 0.05

    quanta, phase = cell_field()
    store = new_store((6, 6, 6), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(store, quanta, phase)
    before = memory_per_box(store)
    switch_levels(store)
    after = memory_per_box(store)
    filled = 4 ** 3 / 6 ** 3
    assert np.isclose(before["in_use"], filled * 8.0 + 2 * 4 / 512)
    assert np.isclose(after["in_use"], filled * 1.75 + 2 * 4 / 512)

    tiled = new_store((4, 4, 4), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(tiled, np.tile(np.full((8, 8, 8), -700, np.int32), (4, 4, 4)), np.zeros((32, 32, 32), np.float32))
    switch_levels(tiled)
    shared = memory_per_box(tiled)
    assert shared["blocks_reused"] == 64 and np.isclose(shared["reused"], 8.0 / 64)
    print(f"\nbytes per box: full {before['full']}, coarse {after['coarse']}, reused {shared['reused']} "
          f"(+{shared['reused_host']} host key) for 64 sharing blocks, empty {blank['empty']} + index {blank['index']:.4f}; "
          f"cell field in use {before['in_use']:.3f} -> {after['in_use']:.3f}")
