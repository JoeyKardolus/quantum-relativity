"""Stepper: steps a source on its own thread and hands the window snapshots to draw.

Contract: `Stepper(source, steps_per_frame, paused=False)` starts a daemon thread that
publishes the start, then runs `steps_per_frame` source steps, then publishes a
Snapshot, and repeats while not paused. A Snapshot carries the step count, the
triangles to draw (positions float32 (n, 3), colours uint8 (n, 4), `grid_triangles` of
the source's `view()`; the source interface is in `cells_source.py`), the cells drawn,
the grid's bounding corners low and high, the latest stats (refreshed at most every
STATS_EVERY seconds, and on every reset, pause or change of view), steps per second,
paused, error, generation (bumped on each reset, so the window can re-fit) and layer
(the slice's lattice index along z, None for the whole grid). It starts on the slice
at `middle_layer`. `latest()` returns the newest Snapshot, or None when none is new
since the last call. `show(kinds)` picks which kinds are drawn (`grid_triangles`
SHOW_ENERGY_VOLUMES, SHOW_JOINTS or SHOW_BOTH, the start; any other value raises
ValueError). `toggle_whole()` switches between the slice and the whole grid.
`move_slice(layers)` moves the slice by whole lattice layers, round the wrapping grid
(nothing while the whole grid shows). Each of these republishes even while paused.
`toggle_pause()`, `show()`, `toggle_whole()`, `move_slice()`, `reset()` and `stop()`
are safe from any thread. All source work (steps, views) runs on the stepping thread.
When the source or the layout raises, stepping stops and the error is in the snapshot
(its traceback printed); nothing is swallowed.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, replace

import numpy as np

from physics.view.grid_triangles import SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS, grid_triangles
from physics.view.middle_layer import middle_layer

STATS_EVERY = 0.5


@dataclass(frozen=True)
class Snapshot:
    """What the window draws and titles: one moment of the source."""

    step: int
    positions: np.ndarray
    colours: np.ndarray
    drawn: int
    low: np.ndarray
    high: np.ndarray
    stats: dict
    steps_per_second: float
    paused: bool
    error: str
    generation: int
    layer: int | None


class Stepper:
    """Runs a source on a thread; the window reads its snapshots."""

    def __init__(self, source, steps_per_frame: int = 1, paused: bool = False) -> None:
        self.source = source
        self.whole = False
        self.slice_offset = 0
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.paused = paused
        self.shown = SHOW_BOTH
        self.error = ""
        self.generation = 0
        self._reset_wanted = threading.Event()
        self._stop_wanted = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._stats = {}
        self._stats_at = 0.0
        self._rate = 0.0
        self._last = None
        self._thread = threading.Thread(target=self._run, name="stepper", daemon=True)
        self._thread.start()

    def latest(self) -> Snapshot | None:
        """The newest snapshot not handed out yet, or None."""
        with self._lock:
            snapshot, self._latest = self._latest, None
        return snapshot

    def toggle_pause(self) -> None:
        """Pause or resume stepping."""
        self.paused = not self.paused
        self._stats_at = 0.0

    def show(self, kinds: int) -> None:
        """Draw only energy volumes, only joints, or both, from the next snapshot on."""
        if kinds not in (SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH):
            raise ValueError(f"kinds: {kinds}, expected {SHOW_ENERGY_VOLUMES}, {SHOW_JOINTS} or {SHOW_BOTH}")
        self.shown = kinds
        self._stats_at = 0.0

    def toggle_whole(self) -> None:
        """Draw the whole grid, or the slice again."""
        self.whole = not self.whole
        self._stats_at = 0.0

    def move_slice(self, layers: int) -> None:
        """Move the slice by whole lattice layers; nothing while the whole grid shows."""
        if not self.whole:
            self.slice_offset += int(layers)
            self._stats_at = 0.0

    def reset(self) -> None:
        """Ask for a fresh start on the stepping thread."""
        self._reset_wanted.set()

    def stop(self) -> None:
        """Stop the thread and wait for it."""
        self._stop_wanted.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        self._guarded(self._publish)
        while not self._stop_wanted.is_set():
            if self._reset_wanted.is_set():
                self._reset_wanted.clear()
                self.error, self._stats_at = "", 0.0
                self.generation += 1
                self._guarded(self._fresh_start)
            elif self.paused or self.error:
                if self._stats_at == 0.0 and not self.error:
                    self._guarded(self._publish)
                self._stop_wanted.wait(0.05)
            else:
                self._guarded(self._advance)

    def _fresh_start(self) -> None:
        self.source.reset()
        self._publish()

    def _advance(self) -> None:
        started = time.perf_counter()
        for _ in range(self.steps_per_frame):
            self.source.step()
        self._publish()
        self._rate = self.steps_per_frame / max(time.perf_counter() - started, 1e-9)

    def _guarded(self, action) -> bool:
        """Run one action; when it raises, print it, stop stepping and show the error."""
        try:
            action()
            return True
        except Exception as raised:
            traceback.print_exc()
            self.error = f"{type(raised).__name__}: {raised}"
            with self._lock:
                if self._last is not None:
                    self._latest = replace(self._last, error=self.error, paused=True)
                    self._last = self._latest
            return False

    def _publish(self) -> None:
        pieces = self.source.view()
        layer = None if self.whole else (middle_layer(pieces.side) + self.slice_offset) % pieces.side
        positions, colours, drawn = grid_triangles(pieces, self.shown, layer)
        now = time.perf_counter()
        if now - self._stats_at >= STATS_EVERY:
            self._stats, self._stats_at = self.source.stats(), now
        snapshot = Snapshot(step=int(self.source.steps), positions=positions, colours=colours, drawn=drawn,
                            low=pieces.low, high=pieces.high, stats=self._stats, steps_per_second=self._rate,
                            paused=self.paused, error=self.error, generation=self.generation, layer=layer)
        with self._lock:
            self._latest = self._last = snapshot
