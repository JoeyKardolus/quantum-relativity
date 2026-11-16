"""Stepper: steps a source on its own thread and hands the window snapshots to draw.

Contract: `Stepper(source, steps_per_frame, paused=False, look=NEAREST)` starts a daemon thread
that publishes the start, then runs `steps_per_frame` source steps,
then publishes a Snapshot, and repeats while not paused. A Snapshot carries the
step count, the shown boxes' positions (float32 (n, 3)) and colours (uint8
(n, 4)) gathered on the GPU (`BoxPoints`) from the source's `view()` (the source
interface is in `cells_source.py`), the count of boxes with negative energy, the
occupied boxes' bounding corners low and high, the latest stats (refreshed at most every STATS_EVERY seconds, and on
every reset or pause), steps per second, paused, error and generation (bumped on
each reset, so the window can re-fit). The light reference (`start_reference`)
is taken at the start and on each reset, then held fixed. `latest()` returns the
newest Snapshot, or None when none is new since the last call. `show(kinds)`
picks which kinds are drawn (`box_points` SHOW_ENERGY_VOLUMES, SHOW_JOINTS or
SHOW_BOTH, the start; a snapshot is published again even while paused; any other
value raises ValueError). `draw_as(look)` picks how they are lit (`box_points`
NEAREST, the start, or GLOW; any other value raises ValueError). `toggle_slice()`
turns the one-layer slice through the middle on (at the middle layer) or off;
`toggle_slab()` turns the slab on or off: SLAB_BOXES layers through the middle,
drawn in the SLAB look whatever `draw_as` chose (the slice and the slab exclude
each other; turning one on turns the other off). `move_slice(layers)` moves
whichever is on by whole box layers (held inside the occupied layers; nothing
while both are off). Each of these republishes even while paused, and the
snapshot carries look (SLAB while the slab is on) and slice_at (the centre layer,
NaN without a slice or slab).
`toggle_pause()`, `show()`, `draw_as()`, `toggle_slice()`, `toggle_slab()`, `move_slice()`,
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

from physics.view.box_points import (GLOW, NEAREST, SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS,
                                     SLAB, SLAB_BOXES, SLICE_BOXES, BoxPoints)
from physics.view.start_reference import start_reference

STATS_EVERY = 0.5


@dataclass(frozen=True)
class Snapshot:
    """What the window draws and titles: one moment of the source."""

    step: int
    positions: np.ndarray
    colours: np.ndarray
    negative: int
    low: np.ndarray
    high: np.ndarray
    stats: dict
    steps_per_second: float
    paused: bool
    error: str
    generation: int
    look: int = NEAREST
    slice_at: float = float("nan")


class Stepper:
    """Runs a source on a thread; the window reads its snapshots."""

    def __init__(self, source, steps_per_frame: int = 1, paused: bool = False,
                 look: int = NEAREST) -> None:
        if look not in (GLOW, NEAREST):
            raise ValueError(f"look: {look}, expected {GLOW} (glow) or {NEAREST} (nearest)")
        self.source = source
        self.look = look
        self.slicing = False
        self.slab = False
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

    def show(self, kinds: int) -> None:
        """Draw only energy volumes, only joints, or both, from the next snapshot on."""
        if kinds not in (SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH):
            raise ValueError(f"kinds: {kinds}, expected {SHOW_ENERGY_VOLUMES}, {SHOW_JOINTS} or {SHOW_BOTH}")
        self.shown = kinds
        self._stats_at = 0.0

    def draw_as(self, look: int) -> None:
        """Light the boxes as NEAREST (depth-tested, by kind) or GLOW (additive), from the next snapshot on."""
        if look not in (GLOW, NEAREST):
            raise ValueError(f"look: {look}, expected {GLOW} (glow) or {NEAREST} (nearest)")
        self.look = look
        self._stats_at = 0.0

    def toggle_slice(self) -> None:
        """Draw only the slice through the middle layer, or the whole space again."""
        self.slicing = not self.slicing
        self.slab = False
        self.slice_offset = 0
        self._stats_at = 0.0

    def toggle_slab(self) -> None:
        """Draw only the SLAB_BOXES-thick slab through the middle in the SLAB look, or the whole space again."""
        self.slab = not self.slab
        self.slicing = False
        self.slice_offset = 0
        self._stats_at = 0.0

    def move_slice(self, layers: int) -> None:
        """Move the slice or slab by whole box layers; nothing while both are off."""
        if self.slicing or self.slab:
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
        view = self.source.view()
        if self._reference is None:
            self._reference = start_reference(view.density, view.kind)
        cutting = self.slicing or self.slab
        asked = self.slice_offset if cutting else None
        look = SLAB if self.slab else self.look
        thickness = SLAB_BOXES if self.slab else SLICE_BOXES
        gathered = self._points.boxes(view, self._reference, self.shown, look, asked, thickness)
        if gathered.slice_offset is not None and cutting and self.slice_offset == asked:
            self.slice_offset = gathered.slice_offset  # held inside the occupied layers
        now = time.perf_counter()
        if now - self._stats_at >= STATS_EVERY:
            self._stats, self._stats_at = self.source.stats(), now
        snapshot = Snapshot(step=int(self.source.steps),
                            positions=gathered.positions, colours=gathered.colours,
                            negative=gathered.negative, low=gathered.low, high=gathered.high,
                            stats=self._stats,
                            steps_per_second=self._rate, paused=self.paused,
                            error=self.error, generation=self.generation,
                            look=look, slice_at=gathered.slice_at)
        with self._lock:
            self._latest = self._last = snapshot
