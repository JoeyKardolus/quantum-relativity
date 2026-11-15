"""Stepper: steps a source on its own thread and hands the window snapshots to draw.

Contract: `Stepper(source, steps_per_frame, paused=False)` starts a daemon thread
that publishes the start, then runs `steps_per_frame` source steps (the source interface is in `fluid_source.py`),
then publishes a Snapshot, and repeats while not paused. A Snapshot carries the
step count, side, the shown boxes' positions (float32 (n, 3)) and colours (uint8
(n, 4)) gathered on the GPU (`BoxPoints`), the count of boxes with negative
energy, the latest stats (refreshed at most every STATS_EVERY seconds, and on
every reset or pause), steps per second, paused, error and generation (bumped on
each reset, so the window can re-fit). The light reference (`start_reference`)
is taken at the start and on each reset, then held fixed. `latest()` returns the
newest Snapshot, or None when none is new since the last call. `toggle_pause()`,
`reset()` and `stop()` are safe from any thread. All GPU work (steps, gathers)
runs on the stepping thread. When the source or the gather raises, stepping stops
and the error is in the snapshot (its traceback printed); nothing is swallowed.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, replace

import numpy as np

from physics.view.box_points import BoxPoints
from physics.view.start_reference import start_reference

STATS_EVERY = 0.5


@dataclass(frozen=True)
class Snapshot:
    """What the window draws and titles: one moment of the source."""

    step: int
    side: int
    positions: np.ndarray
    colours: np.ndarray
    negative: int
    stats: dict
    steps_per_second: float
    paused: bool
    error: str
    generation: int


class Stepper:
    """Runs a source on a thread; the window reads its snapshots."""

    def __init__(self, source, steps_per_frame: int = 1, paused: bool = False) -> None:
        self.source = source
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.paused = paused
        self.error = ""
        self.generation = 0
        self._reset_wanted = threading.Event()
        self._stop_wanted = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._stats = {}
        self._stats_at = 0.0
        self._rate = 0.0
        self._reference = None
        self._last = None
        self._points = BoxPoints()
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
                self.error, self._reference, self._stats_at = "", None, 0.0
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
        if self._reference is None:
            self._reference = start_reference(self.source.a_energy, self.source.b_energy)
        gathered = self._points.boxes(self.source.a_energy, self.source.b_energy, self._reference)
        now = time.perf_counter()
        if now - self._stats_at >= STATS_EVERY:
            self._stats, self._stats_at = self.source.stats(), now
        snapshot = Snapshot(step=int(self.source.steps), side=int(self.source.side),
                            positions=gathered.positions, colours=gathered.colours,
                            negative=gathered.negative, stats=self._stats,
                            steps_per_second=self._rate, paused=self.paused,
                            error=self.error, generation=self.generation)
        with self._lock:
            self._latest = self._last = snapshot
