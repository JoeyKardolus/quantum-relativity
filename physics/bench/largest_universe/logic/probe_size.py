"""Probe size: build one size, time it, and let it go.

Contract: `probe_size(build, size, steps, warmup)` calls `build(size)`, which returns a
`Run` or raises MemoryError when the size does not fit. A fitting run is timed with
`throughput`, its facts are read, and the universe is dropped and the device
synchronized so probes do not stack in memory. Returns a `Probe`; a MemoryError gives
`Probe(size, fits=False)`. Any other error from build or a step is raised as it is.
"""
from __future__ import annotations

import gc
from typing import Callable

import warp as wp

from physics.bench.largest_universe.definitions.search import Probe
from physics.bench.throughput.definitions.run import Run
from physics.bench.throughput.logic.throughput import throughput


def probe_size(build: Callable[[int], Run], size: int, steps: int, warmup: int) -> Probe:
    """One size as a Probe: timed when it fits, marked when it does not."""
    try:
        run = build(size)
    except MemoryError:
        return Probe(size, fits=False)
    rates = throughput(run.advance, run.cells, steps, warmup)
    probe = Probe(size, True, run.cells, rates.steps_per_second, run.facts())
    del run
    gc.collect()
    wp.synchronize()
    return probe
