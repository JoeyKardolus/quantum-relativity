"""Stepper: steps a source on its own thread and hands the window snapshots to draw.

Contract: `Stepper(source, steps_per_frame)` starts a daemon thread that runs
`steps_per_frame` source steps, then publishes a Snapshot, and repeats while not
paused. A Snapshot carries the step count, positions (float32, (n, 3)), colours
(`point_colours`, uint8 (n, 4)), the latest stats (refreshed at most every
STATS_EVERY seconds, and on every reset or pause), steps per second, paused,
error and generation (bumped on each reset, so the window can re-centre).
The colour reference (`density_reference`) is taken at the start and on each
reset, then held fixed. `latest()` returns the newest Snapshot, or None when none
is new since the last call. `toggle_pause()`, `reset()` and `stop()` are safe from
any thread. When the source raises, stepping stops and the error is in the
snapshot (and its traceback is printed); nothing is swallowed.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, replace

import numpy as np

from physics.view.density_reference import density_reference
from physics.view.point_colours import point_colours

STATS_EVERY = 0.5


@dataclass(frozen=True)
class Snapshot:
    """What the window draws and titles: one moment of the source."""

    step: int
    positions: np.ndarray
    colours: np.ndarray
    stats: dict
    steps_per_second: float
    paused: bool
    error: str
    generation: int


class Stepper:
    """Runs a source on a thread; the window reads its snapshots."""

    def __init__(self, source, steps_per_frame: int = 1) -> None:
        self.source = source
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.paused = False
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
        self._rate = self.steps_per_frame / max(time.perf_counter() - started, 1e-9)
        self._publish()

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
        kind = _host(self.source.kind)
        density = _host(self.source.density)
        if self._reference is None:
            self._reference = density_reference(kind, density)
        now = time.perf_counter()
        if now - self._stats_at >= STATS_EVERY:
            self._stats, self._stats_at = self.source.stats(), now
        snapshot = Snapshot(step=int(self.source.steps),
                            positions=np.ascontiguousarray(_host(self.source.positions), dtype=np.float32),
                            colours=point_colours(kind, density, self._reference),
                            stats=self._stats, steps_per_second=self._rate,
                            paused=self.paused, error=self.error, generation=self.generation)
        with self._lock:
            self._latest = self._last = snapshot


def _host(array) -> np.ndarray:
    return array.numpy() if hasattr(array, "numpy") else np.asarray(array)
