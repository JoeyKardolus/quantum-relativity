"""Totals: the whole quanta held in the boxes, counted exactly on the GPU.

Contract: `totals(state)` returns {"amount", "plus", "minus", "boxes", "universes"}:
the sum of |quanta| over the whole batch (the conserved amount, bible rule 4), the plus
quanta (pulling energy), the minus quanta as a positive count (pushing energy) and the
occupied boxes, as Python ints, and under "universes" the same four numbers for each
universe of the batch (state.py), a list of dicts in universe order. Summed with int64
atomics, so every count is exact, never rounded.
"""
from __future__ import annotations

import warp as wp

from physics.engine.state import State


@wp.kernel
def _count(quanta: wp.array3d(dtype=wp.int32), side: int, sums: wp.array2d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    amount = quanta[ti, tj, tk]
    home = ti // side
    if amount > 0:
        wp.atomic_add(sums, home, 0, wp.int64(amount))
        wp.atomic_add(sums, home, 2, wp.int64(1))
    elif amount < 0:
        wp.atomic_add(sums, home, 1, wp.int64(-amount))
        wp.atomic_add(sums, home, 2, wp.int64(1))


def totals(state: State) -> dict:
    """The exact plus, minus and total quanta and the occupied box count, overall and per universe."""
    sums = wp.zeros((state.universes, 3), dtype=wp.int64, device=state.quanta.device)
    wp.launch(_count, dim=state.shape, device=state.quanta.device, inputs=[state.quanta, state.side, sums])
    each = [_named(*(int(value) for value in row)) for row in sums.numpy()]
    whole = _named(sum(row["plus"] for row in each), sum(row["minus"] for row in each),
                   sum(row["boxes"] for row in each))
    return whole | {"universes": each}


def _named(plus: int, minus: int, boxes: int) -> dict:
    return {"amount": plus + minus, "plus": plus, "minus": minus, "boxes": boxes}
