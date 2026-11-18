"""Stepper: steps a source on its own thread and hands the window snapshots to draw.

Contract: `Stepper(source, steps_per_frame=1, paused=False)` starts a daemon thread that
publishes the start, then runs `steps_per_frame` source steps, publishes a `Snapshot`,
and repeats while not paused (`source` fits the `Source` interface). It starts on the
slice through `middle_layer`: only that layer is laid out and drawn, whatever the size
of the lattice. With the whole grid on, the packed cells are laid out too and handed
over as blocks beside the slice. Stats are read at most every STATS_EVERY seconds, and
on every reset, pause or change of view.
  latest()         the newest Snapshot, or None when none is new since the last call
  toggle_pause()   pause or resume
  show(kinds)      draw only energy volumes, only joints, or both (one of SHOWN; any
                   other value raises ValueError)
  toggle_whole()   the whole grid (the slice plus the packed cells' blocks), or the
                   slice alone
  move_slice(n)    move the slice by whole lattice layers, round the wrapping grid
  reset(), stop()  a fresh start; stop the thread and wait for it
Each of these is safe from any thread and republishes even while paused. All source
work (steps, views) runs on the stepping thread. When the source or the layout raises,
stepping stops and the error is in the snapshot (its traceback printed); nothing is
swallowed.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import replace

from physics.view.drawing.definitions.middle_layer import middle_layer
from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH, SHOWN
from physics.view.drawing.logic.grid_triangles import grid_triangles
from physics.view.window.definitions.snapshot import Snapshot
from physics.view.window.definitions.window_marks import IDLE_WAIT, STATS_EVERY


class Stepper:
    """Runs a source on a thread; the window reads its snapshots."""

    def __init__(self, source, steps_per_frame: int = 1, paused: bool = False) -> None:
        self.source = source
        self.whole = False
        self._side = None
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
        if kinds not in SHOWN:
            raise ValueError(f"kinds: {kinds}, expected one of {SHOWN}")
        self.shown = kinds
        self._stats_at = 0.0

    def toggle_whole(self) -> None:
        """Draw the whole grid, or the slice alone again."""
        self.whole = not self.whole
        self._stats_at = 0.0

    def move_slice(self, layers: int) -> None:
        """Move the slice by whole lattice layers."""
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
                self._stop_wanted.wait(IDLE_WAIT)
            else:
                self._guarded(self._advance)

    def _fresh_start(self) -> None:
        self.source.reset()
        self._side = None
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
        if self._side is None:
            self._side = self.source.view(0).side
        layer = (middle_layer(self._side) + self.slice_offset) % self._side
        pieces = self.source.view(layer)
        corners, colours, pattern, drawn = grid_triangles(pieces, self.shown, layer)
        blocks, limited, left_out = None, pieces.limited, 0
        if self.whole:
            packed = self.source.view()
            *blocks, more = grid_triangles(packed, self.shown, None)
            blocks, drawn, limited, left_out = tuple(blocks), drawn + more, limited or packed.limited, packed.left_out
        now = time.perf_counter()
        if now - self._stats_at >= STATS_EVERY:
            self._stats, self._stats_at = self.source.stats(), now
        snapshot = Snapshot(step=int(self.source.steps), corners=corners, colours=colours, pattern=pattern, drawn=drawn,
                            low=pieces.low, high=pieces.high, stats=self._stats, steps_per_second=self._rate,
                            paused=self.paused, error=self.error, generation=self.generation, layer=layer,
                            blocks=blocks, places=self._side, limited=limited, left_out=left_out)
        with self._lock:
            self._latest = self._last = snapshot
