"""Amounts: each universe's plus amount and minus amount, counted exactly on the device.

Contract: `amounts(batch)` returns two host int64 numpy arrays of length universe_count,
(plus, minus): plus[u] the sum of the positive quanta in universe u, minus[u] the sum of
the magnitudes of its negative quanta, so plus + minus is its total amount ignoring
sign (bible rule 4: the conserved quantity). Integer sums, exact; synchronizes the device.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.bench.quanta_batch import QuantaBatch


@wp.kernel
def _tally(quanta: wp.array4d(dtype=wp.int32), plus: wp.array(dtype=wp.int64), minus: wp.array(dtype=wp.int64)):
    universe, ix, iy, iz = wp.tid()
    held = quanta[universe, ix, iy, iz]
    if held > 0:
        wp.atomic_add(plus, universe, wp.int64(held))
    if held < 0:
        wp.atomic_add(minus, universe, wp.int64(-held))


def amounts(batch: QuantaBatch) -> tuple[np.ndarray, np.ndarray]:
    """Per-universe plus amount and minus amount, exact int64."""
    device = batch.quanta.device
    plus = wp.zeros(batch.universe_count, dtype=wp.int64, device=device)
    minus = wp.zeros(batch.universe_count, dtype=wp.int64, device=device)
    wp.launch(_tally, dim=tuple(batch.quanta.shape), inputs=[batch.quanta, plus, minus], device=device)
    return plus.numpy(), minus.numpy()
