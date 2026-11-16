"""Clock rate (Warp func): how fast a cell's local clock runs, set by its own packing.

Contract: `clock_rate(packing, clock_packing)` returns 1 / (1 + packing / clock_packing),
in (0, 1]: the more packed a cell (plus or minus: joints count, bible "Time is local"),
the slower everything in it happens. clock_packing <= 0 switches clocks off (rate 1).
cell_laws.py computes the same rate per cell on the host; a start's boxes read it from
their cell.
"""
from __future__ import annotations

import warp as wp


@wp.func
def clock_rate(packing: float, clock_packing: float) -> float:
    """The clock rate: 1 at no packing, 1/2 at clock_packing quanta per box."""
    if clock_packing <= 0.0:
        return 1.0
    return 1.0 / (1.0 + packing / clock_packing)
