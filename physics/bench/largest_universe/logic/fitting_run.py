"""Fitting run: an engine universe of a size, refused before it is built when it will not fit.

Contract: `fitting_run(cells_per_side, free_bytes=None)` returns `engine_run(cells_per_side)`
when the memory law says the size fits `free_bytes`, and raises MemoryError otherwise,
before anything is allocated, so no probe spills into WSL's shared system memory.
`free_bytes=None` reads the GPU's free memory now; other programs on the card shrink it.
"""
from __future__ import annotations

import warp as wp

from physics.bench.largest_universe.definitions.memory_law import bytes_needed, fits
from physics.bench.throughput.definitions.engine_start import DEVICE
from physics.bench.throughput.definitions.run import Run
from physics.bench.throughput.logic.engine_run import engine_run


def fitting_run(cells_per_side: int, free_bytes: float | None = None) -> Run:
    """The engine's Run of this size, or MemoryError when its bytes exceed the free bytes."""
    free = wp.get_device(DEVICE).free_memory if free_bytes is None else free_bytes
    if not fits(cells_per_side, free):
        raise MemoryError(f"{cells_per_side} cells a side needs {bytes_needed(cells_per_side):,.0f} bytes, "
                          f"{free:,.0f} are free")
    return engine_run(cells_per_side)
