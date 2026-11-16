"""Settle counts: apply one phase's box-count changes at once, so the boxes move the same whatever order the GPU runs them in.

Contract: `settle_counts(boxes, counts)` takes the per-cell box counts (int32, (universes,
capacity)) and the phase's three count arrays [ins, outs, delta] of the same shape:
exchange.py and mortar_shift.py first propose every move of a phase (ins: boxes
proposed into each cell, outs: out of it), then commit only the moves that keep the
cells' order even if every proposal went through, adding what moved to delta. This adds
delta to boxes and zeroes all three. Integer sums only, so the result does not depend on
the order the GPU adds them in (numerics: before 2026-11-16 a move was put back when
other moves of the same phase had raced ahead, and two runs of one seed differed).
"""
from __future__ import annotations

import warp as wp


@wp.kernel
def _settle(boxes: wp.array2d(dtype=wp.int32), ins: wp.array2d(dtype=wp.int32), outs: wp.array2d(dtype=wp.int32),
            delta: wp.array2d(dtype=wp.int32)):
    home, cell = wp.tid()
    boxes[home, cell] = boxes[home, cell] + delta[home, cell]
    ins[home, cell] = 0
    outs[home, cell] = 0
    delta[home, cell] = 0


def settle_counts(boxes: wp.array, counts: list) -> None:
    """boxes += delta; ins, outs and delta zeroed."""
    wp.launch(_settle, dim=boxes.shape, device=boxes.device, inputs=[boxes, *counts])
