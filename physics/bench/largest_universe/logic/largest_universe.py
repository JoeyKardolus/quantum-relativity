"""Largest universe: the biggest size of a list that still runs at the asked rate.

Contract: `largest_universe(build, sizes=SIZES, min_rate=MIN_RATE, steps=STEPS,
warmup=WARMUP)` bisects the rising list `sizes`, taking a bigger size to be never
faster, so it probes at most log2(len) + 1 sizes. A size passes when it fits and
`meets_rate`. Returns a `Search` (search.py): the largest passing size with its cells
and rate, every probe in order, and the limit. Raises ValueError for a list
`check_sizes` refuses or a rate `check_rate` refuses, before anything is built.
"""
from __future__ import annotations

from typing import Callable, Sequence

from physics.bench.largest_universe.definitions.limits import limit_law
from physics.bench.largest_universe.definitions.pass_mark import MIN_RATE, STEPS, WARMUP, check_rate, meets_rate
from physics.bench.largest_universe.definitions.search import Search
from physics.bench.largest_universe.definitions.size_ladder import SIZES, check_sizes
from physics.bench.largest_universe.logic.probe_size import probe_size
from physics.bench.throughput.definitions.run import Run


def largest_universe(build: Callable[[int], Run], sizes: Sequence[int] = SIZES, min_rate: float = MIN_RATE,
                     steps: int = STEPS, warmup: int = WARMUP) -> Search:
    """Bisect `sizes` for the largest that fits and runs at `min_rate` steps a second."""
    check_sizes(sizes)
    check_rate(min_rate)
    ordered = list(sizes)
    probes = []
    best = None
    passing = -1
    failing = len(ordered)
    while failing - passing > 1:
        middle = (passing + failing) // 2
        probe = probe_size(build, ordered[middle], steps, warmup)
        probes.append(probe)
        if probe.fits and meets_rate(probe.steps_per_second, min_rate):
            passing = middle
            best = probe
        else:
            failing = middle
    limit = limit_law(probes, min_rate)
    if best is None:
        return Search(None, None, None, tuple(probes), limit)
    return Search(best.size, best.cells, best.steps_per_second, tuple(probes), limit)
