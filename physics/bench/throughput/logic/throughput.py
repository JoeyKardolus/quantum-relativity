"""Throughput: time a universe's steps on the device.

Contract: `throughput(advance, cells, steps=STEPS, warmup=WARMUP)` calls `advance()`
`warmup` times untimed, then `steps` times between two device synchronizations, and
returns the `Rates` of those timed steps (rates.py). `advance` is any callable that takes
one universe one step; nothing here knows the engine. Raises ValueError for a plan
`check_timing` refuses, before advance is called at all.
"""
from __future__ import annotations

import time
from typing import Callable

import warp as wp

from physics.bench.throughput.definitions.rates import Rates, rates_law
from physics.bench.throughput.definitions.timing import STEPS, WARMUP, check_timing


def throughput(advance: Callable[[], None], cells: int, steps: int = STEPS, warmup: int = WARMUP) -> Rates:
    """Time `steps` calls of advance after `warmup` untimed ones."""
    check_timing(cells, steps, warmup)
    for _ in range(warmup):
        advance()
    wp.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        advance()
    wp.synchronize()
    return rates_law(cells, steps, time.perf_counter() - started)
