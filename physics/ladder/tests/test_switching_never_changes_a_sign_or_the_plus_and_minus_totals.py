"""Many switch and promote passes on wild random fields: every sign and both totals exact, every pass.

The error bounds are opened wide here (max_rel_error 1e9, max_phase_error 10) so every
quiet block is coded COARSE however rough: this is the worst case for exactness, not a
setting. Magnitudes up to a million, mixed signs and zeros in one block, a few all-zero
blocks. Group sums per sign are checked too, so no quantum moves between groups.
"""
from __future__ import annotations

import numpy as np

from physics.ladder import Level, LadderSettings, load_region, new_store, promote_blocks, read_region
from physics.ladder import switch_levels, total_quanta
from physics.ladder.tests._fields import DEVICE


def _group_sums(quanta: np.ndarray) -> np.ndarray:
    """Plus and minus sums of every 4^3 group."""
    shaped = quanta.astype(np.int64).reshape(quanta.shape[0] // 4, 4, quanta.shape[1] // 4, 4, quanta.shape[2] // 4, 4)
    return np.stack([np.where(shaped > 0, shaped, 0).sum(axis=(1, 3, 5)), np.where(shaped < 0, -shaped, 0).sum(axis=(1, 3, 5))])


def test_switching_never_changes_a_sign_or_the_plus_and_minus_totals():
    rng = np.random.default_rng(7)
    shape = (32, 32, 16)
    quanta = (rng.integers(1, 1_000_000, size=shape) * rng.choice([-1, 0, 1], size=shape, p=[0.45, 0.1, 0.45]))
    quanta = quanta.astype(np.int32)
    quanta[:8, :8, :8] = 0
    quanta[8:16, :8, :8] = rng.integers(1, 3, size=(8, 8, 8))
    phase = np.full(shape, 0.4, dtype=np.float32)
    store = new_store((4, 4, 2), LadderSettings(quiet_steps=1, max_rel_error=1e9, max_phase_error=10.0), device=DEVICE)
    load_region(store, quanta, phase)
    expected = (int(quanta[quanta > 0].sum()), int(-quanta[quanta < 0].sum()))
    sums = _group_sums(quanta)
    coarse_seen = 0
    for _ in range(6):
        report = switch_levels(store)
        coarse_seen = max(coarse_seen, report["coarse"])
        assert total_quanta(store) == expected
        read_q, _ = read_region(store)
        assert np.array_equal(np.sign(read_q), np.sign(quanta))
        assert np.array_equal(_group_sums(read_q), sums)
        filled = np.argwhere(store.level != Level.EMPTY)
        promote_blocks(store, filled[rng.random(len(filled)) < 0.5])
        assert total_quanta(store) == expected
    assert coarse_seen == 4 * 4 * 2 - 1
