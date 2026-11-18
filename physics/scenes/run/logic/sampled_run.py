"""Sampled run: step a universe and measure it at fixed intervals.

Contract: `sampled_run(universe, settings, steps, every, measure)` measures the universe
once before stepping, then calls physics.engine.step `steps` times, measuring after
every `every`-th step and after the last. `measure(universe)` returns anything; the
result is a list of (time, measurement) pairs, time = steps taken * settings.step_time
(the counter's time, not a cell's). The universe (a physics.engine Grid) is stepped in
place. Raises ValueError for steps < 0 or every < 1; the engine's own errors propagate.
"""
from __future__ import annotations

from typing import Any, Callable

from physics.engine import Grid, Settings, step


def sampled_run(universe: Grid, settings: Settings, steps: int, every: int,
                measure: Callable[[Grid], Any]) -> list[tuple[float, Any]]:
    """(time, measure(universe)) at the start, every `every` steps and at the end."""
    if steps < 0 or every < 1:
        raise ValueError(f"need steps >= 0 and every >= 1, got {steps} and {every}")
    samples = [(0.0, measure(universe))]
    for taken in range(1, steps + 1):
        step(universe, settings)
        if taken % every == 0 or taken == steps:
            samples.append((taken * settings.step_time, measure(universe)))
    return samples
