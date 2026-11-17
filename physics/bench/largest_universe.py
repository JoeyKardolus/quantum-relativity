"""Largest universe: the biggest size of a list that still runs at least min_rate steps a second.

Contract: `largest_universe(build, sizes, min_rate=1.0, steps=3, warmup=1)` searches the
ascending list `sizes` by bisection, assuming a bigger size is never faster. For a size,
`build(size)` returns `(advance, cells)`: a callable advancing one universe by one step
and its cell count; a MemoryError from build marks the size too big (it does not
fit). The probe is timed with `throughput` and passes at steps_per_second >= min_rate.
Returns a dict:
  size              the largest passing size, or None when the smallest fails
  cells             its cell count (None with it)
  steps_per_second  its measured rate (None with it)
  probes            every probe in order: {size, cells, steps_per_second, fits}
Each probe's universe is dropped and the device synchronized before the next, so probes
do not stack in memory. Raises ValueError for an empty or unsorted list, min_rate <= 0.
"""
from __future__ import annotations

import gc
from typing import Callable, Sequence

import warp as wp

from physics.bench.throughput import throughput


def largest_universe(build: Callable[[int], tuple[Callable[[], None], int]], sizes: Sequence[int],
                     min_rate: float = 1.0, steps: int = 3, warmup: int = 1) -> dict:
    """Bisect sizes for the largest whose one-universe step rate is at least min_rate."""
    ordered = list(sizes)
    if not ordered or ordered != sorted(ordered) or min_rate <= 0:
        raise ValueError(f"need an ascending non-empty size list and min_rate > 0, got {ordered}, {min_rate}")
    probes = []
    passing = -1
    failing = len(ordered)
    while failing - passing > 1:
        middle = (passing + failing) // 2
        probe = _probe(build, ordered[middle], steps, warmup)
        probes.append(probe)
        if probe["fits"] and probe["steps_per_second"] >= min_rate:
            passing = middle
        else:
            failing = middle
    if passing < 0:
        return {"size": None, "cells": None, "steps_per_second": None, "probes": probes}
    best = next(probe for probe in probes if probe["size"] == ordered[passing])
    return {"size": best["size"], "cells": best["cells"], "steps_per_second": best["steps_per_second"],
            "probes": probes}


def _probe(build: Callable[[int], tuple[Callable[[], None], int]], size: int, steps: int, warmup: int) -> dict:
    """Build one size and time it; a MemoryError from build is a size that does not fit."""
    try:
        advance, cells = build(size)
    except MemoryError:
        return {"size": size, "cells": None, "steps_per_second": None, "fits": False}
    rate = throughput(advance, 1, cells, steps=steps, warmup=warmup)["steps_per_second"]
    del advance
    gc.collect()
    wp.synchronize()
    return {"size": size, "cells": cells, "steps_per_second": rate, "fits": True}
