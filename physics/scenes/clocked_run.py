"""Clocked run: step a universe, sample it, and read every energy volume's own clock (bible: time is local).

Contract: `clocked_run(universe, settings, steps, every, measure)` calls
physics.engine.step `steps` times on a physics.engine State (universe 0), calling
`measure(universe)` at the start, after every `every`-th step and after the last, as
sampled_run.py does. At every sample it also reads each energy volume (physics.engine
cells: a connected group of plus boxes) and its local clock, the mean clock reading of
its boxes (the proper time they lived, physics.engine tick_clocks). The engine keeps no
labels, so volumes are followed from sample to sample by their shared boxes: a volume
continues the earlier volume it shares the most boxes with; when several share one, the
one sharing most continues it and the others start new ones (a piece of a torn volume
starts its own column, its clock reading carried by its boxes). Returns (samples,
local): samples a list of (counter time, measurement) pairs, counter time = steps taken
* settings.step_time; local a float64 array (samples, columns) of each followed
volume's clock reading at each sample. Column v, for v below the volume count at the
end, is the volume physics.engine.cells labels v at the end (observer_cells.py reads
them so); the columns after it are volumes that ended earlier (fell together with
another), their reading held from their last sample on. A volume not there yet is NaN.
The universe is stepped in place. Raises ValueError for steps < 0 or every < 1; the
engine's own errors propagate.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from physics.engine import Settings, State, cells, host_boxes, step


def clocked_run(universe: State, settings: Settings, steps: int, every: int,
                measure: Callable[[State], Any]) -> tuple[list[tuple[float, Any]], np.ndarray]:
    """(time, measure(universe)) samples and each energy volume's local clock reading at every sample."""
    if steps < 0 or every < 1:
        raise ValueError(f"need steps >= 0 and every >= 1, got {steps} and {every}")
    follower = _Follower()
    samples = [(0.0, measure(universe))]
    follower.read(universe)
    for taken in range(1, steps + 1):
        step(universe, settings)
        if taken % every == 0 or taken == steps:
            samples.append((taken * settings.step_time, measure(universe)))
            follower.read(universe)
    return samples, follower.table()


class _Follower:
    """Follows energy volumes across samples by shared boxes and keeps each one's clock readings."""

    def __init__(self):
        self.labels, self.tracks, self.readings, self.count = None, None, [], 0

    def read(self, universe: State) -> None:
        found = cells(host_boxes(universe))
        volumes = found["volumes"]
        labels = np.where(found["owner"] < volumes, found["owner"], -1)
        tracks = np.full(volumes, -1, dtype=np.int64)
        if self.labels is not None and volumes:
            tracks = self._continue(labels, volumes)
        for volume in np.flatnonzero(tracks < 0):
            tracks[volume] = self.count
            self.count += 1
        self.readings.append(dict(zip(tracks.tolist(), found["clock"][:volumes].tolist())))
        self.labels, self.tracks = labels, tracks

    def _continue(self, labels: np.ndarray, volumes: int) -> np.ndarray:
        """Per current volume the earlier track it continues (-1 for a new one)."""
        both = (labels >= 0) & (self.labels >= 0)
        pairs, shared = np.unique(labels[both].astype(np.int64) * (self.tracks.size + 1) + self.labels[both],
                                  return_counts=True)
        now, before = pairs // (self.tracks.size + 1), pairs % (self.tracks.size + 1)
        order = np.argsort(-shared, kind="stable")
        tracks = np.full(volumes, -1, dtype=np.int64)
        taken = set()
        for pick in order:
            track = int(self.tracks[before[pick]])
            if tracks[now[pick]] < 0 and track not in taken:
                tracks[now[pick]] = track
                taken.add(track)
        return tracks

    def table(self) -> np.ndarray:
        """(samples, columns): the end's volumes first in label order, then the ended tracks."""
        final = self.tracks.tolist()
        ended = [track for track in range(self.count) if track not in set(final)]
        local = np.full((len(self.readings), self.count), np.nan)
        for column, track in enumerate(final + ended):
            last = np.nan
            for row, reading in enumerate(self.readings):
                if track in reading:
                    last = reading[track]
                    local[row, column] = last
                elif not np.isnan(last):
                    local[row, column] = last
        return local
