"""Clock rate (Warp func): how fast a box's local clock runs, set by its own energy concentration.

Contract: `clock_rate(quanta, clock_packing)` returns 1 / (1 + |quanta| / clock_packing),
in (0, 1]: the more packed a box (plus or minus: joints count, bible "Time is local"),
the slower everything there happens. clock_packing <= 0 switches clocks off (rate 1).
"""
from __future__ import annotations

import warp as wp


@wp.func
def clock_rate(quanta: int, clock_packing: float) -> float:
    """The box's clock rate: 1 when empty of energy, 1/2 at clock_packing quanta, falling as 1 / packing."""
    if clock_packing <= 0.0:
        return 1.0
    return 1.0 / (1.0 + float(wp.abs(quanta)) / clock_packing)
