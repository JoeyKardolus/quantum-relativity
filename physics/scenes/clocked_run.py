"""Clocked run: step a universe, sample it, and read every cell's own clock (bible: time is local).

Contract: `clocked_run(universe, settings, steps, every, measure)` refreshes the cell sums
once (physics.potts.cell_sums), then calls physics.potts.step `steps` times, calling
`measure(universe)` at the start, after every `every`-th step and after the last, as
sampled_run.py does. At every sample it also reads each cell's local clock,
`universe.cell_time` (the proper time the cell has lived, physics.potts.local_time; a volume
split's pieces start from the torn volume's reading). Returns (samples, local): samples a
list of (counter time, measurement) pairs, counter time = steps taken * settings.step_time;
local a float64 array (samples, cells) of each cell's clock reading at each sample, cells the
cell count at the end. A cell id that did not exist yet at a sample is NaN there; a vanished
cell's reading stops where it was. The universe is stepped in place.
Raises ValueError for steps < 0 or every < 1; the engine's own errors propagate.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from physics.potts import Settings, Universe, cell_sums, step


def clocked_run(universe: Universe, settings: Settings, steps: int, every: int,
                measure: Callable[[Universe], Any]) -> tuple[list[tuple[float, Any]], np.ndarray]:
    """(time, measure(universe)) samples and each cell's local clock reading at every sample."""
    if steps < 0 or every < 1:
        raise ValueError(f"need steps >= 0 and every >= 1, got {steps} and {every}")
    cell_sums(universe, settings)
    samples = [(0.0, measure(universe))]
    rows = [universe.cell_time.numpy().copy()]
    for taken in range(1, steps + 1):
        step(universe, settings)
        if taken % every == 0 or taken == steps:
            samples.append((taken * settings.step_time, measure(universe)))
            rows.append(universe.cell_time.numpy().copy())
    local = np.full((len(rows), rows[-1].size), np.nan)
    for index, row in enumerate(rows):
        local[index, :row.size] = row
    return samples, local
