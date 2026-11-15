"""Sampled run: step a grid and measure it at fixed intervals.

Contract: `sampled_run(grid, settings, steps, every, measure)` measures the grid once
before stepping, then calls physics.fluid.step `steps` times, measuring after every
`every`-th step and after the last. `measure(grid)` returns anything; the result is a
list of (time, measurement) pairs, time = steps taken * settings.step_time. The grid is
stepped in place. Raises ValueError for steps < 0 or every < 1; the engine's own errors
(too many substeps, a non-finite flow) propagate.
"""
from __future__ import annotations

from typing import Any, Callable

from physics.fluid import Grid, Settings, step


def sampled_run(grid: Grid, settings: Settings, steps: int, every: int,
                measure: Callable[[Grid], Any]) -> list[tuple[float, Any]]:
    """(time, measure(grid)) at the start, every `every` steps and at the end."""
    if steps < 0 or every < 1:
        raise ValueError(f"need steps >= 0 and every >= 1, got {steps} and {every}")
    samples = [(0.0, measure(grid))]
    for taken in range(1, steps + 1):
        step(grid, settings)
        if taken % every == 0 or taken == steps:
            samples.append((taken * settings.step_time, measure(grid)))
    return samples
