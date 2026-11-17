"""Throughput: how many universe-steps a second an engine runs, timed on the device.

Contract: `throughput(advance, universe_count, cells_per_universe, steps=20, warmup=2)`
calls `advance()` warmup times untimed (kernel compiles, first-step set-up), then
`steps` times between two device synchronizations, and returns a dict:
  universes                     universe_count
  cells_per_universe            cells_per_universe
  steps                         timed calls of advance
  seconds                       wall time of the timed calls, synchronized at both ends
  steps_per_second              steps / seconds (each call advances every universe one step)
  universe_steps_per_second     universes * steps / seconds
  cell_steps_per_second         universes * cells_per_universe * steps / seconds
`advance` is any callable that advances universe_count universes by one step each (the
engine steps one universe; a loop over several engines counts them all). Raises
ValueError for steps < 1, warmup < 0 or a non-positive count.
"""
from __future__ import annotations

import time
from typing import Callable

import warp as wp


def throughput(advance: Callable[[], None], universe_count: int, cells_per_universe: int, steps: int = 20,
               warmup: int = 2) -> dict:
    """Time `steps` calls of advance after `warmup` untimed ones; rates per second."""
    if steps < 1 or warmup < 0 or universe_count < 1 or cells_per_universe < 1:
        raise ValueError(f"need steps >= 1, warmup >= 0 and positive counts, got {steps}, {warmup}, "
                         f"{universe_count}, {cells_per_universe}")
    for _ in range(warmup):
        advance()
    wp.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        advance()
    wp.synchronize()
    seconds = time.perf_counter() - started
    return {"universes": universe_count, "cells_per_universe": cells_per_universe, "steps": steps,
            "seconds": seconds, "steps_per_second": steps / seconds,
            "universe_steps_per_second": universe_count * steps / seconds,
            "cell_steps_per_second": universe_count * cells_per_universe * steps / seconds}
