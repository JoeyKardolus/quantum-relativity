"""Point window: the GPU window that draws a stepper's snapshots and takes the controls.

Contract: `PointWindow(stepper)` opens a resizable pyglet window (vsync off) that, on
every draw, uploads the stepper's newest snapshot (if any) to a `PointCloud` and
draws it through an `orbit_camera`. On the first snapshot and after each reset
(a new generation) the view is fitted: centre at the mean position, distance so
the 99th-percentile reach fills the view, and a dot about half the mean spacing
across. Controls: mouse drag rotates, wheel zooms, space pauses, R resets the
source (and the view), Esc quits. The title (`title_line`) refreshes 4 times a
second. Closing the window stops the stepper. `run(interval)` runs the pyglet loop,
drawing every `interval` seconds (0: as fast as it can).
"""
from __future__ import annotations

import time

import numpy as np
import pyglet
from pyglet.window import key

from physics.view.orbit_camera import orbit_camera
from physics.view.point_cloud import PointCloud
from physics.view.title_line import title_line

FOV_DEGREES = 45.0
TURN_PER_PIXEL = 0.01
ZOOM_PER_NOTCH = 0.9


class PointWindow(pyglet.window.Window):
    """The window: points from the stepper, an orbit camera, keys and a status title."""

    def __init__(self, stepper, width: int = 1280, height: int = 800) -> None:
        super().__init__(width=width, height=height, resizable=True, vsync=False,
                         caption="quantum-relativity")
        self.stepper = stepper
        self.cloud = PointCloud()
        self.snapshot = None
        self.generation = -1
        self.centre, self.distance, self.point_world = np.zeros(3), 1.0, 0.05
        self.yaw, self.pitch = 0.6, 0.35
        self._frames, self._frames_since = 0, time.perf_counter()
        self.frames_per_second = 0.0
        self._titled_at = 0.0

    def on_draw(self) -> None:
        """Take the newest snapshot, draw it, count the frame."""
        snapshot = self.stepper.latest()
        if snapshot is not None:
            self.cloud.upload(snapshot.positions, snapshot.colours)
            if snapshot.generation != self.generation:
                self._fit(snapshot.positions)
                self.generation = snapshot.generation
            self.snapshot = snapshot
        width, height = self.get_framebuffer_size()
        pyglet.gl.glViewport(0, 0, width, height)
        pyglet.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.clear()
        camera = orbit_camera(self.centre, self.distance, self.yaw, self.pitch,
                              width / max(height, 1), FOV_DEGREES)
        focal = 1.0 / np.tan(np.radians(FOV_DEGREES) / 2.0)
        self.cloud.draw(camera, self.point_world * focal * height / 2.0)
        self._count_frame()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        """Drag turns the view around the centre."""
        self.yaw -= dx * TURN_PER_PIXEL
        self.pitch = float(np.clip(self.pitch - dy * TURN_PER_PIXEL, -1.55, 1.55))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y) -> None:
        """Wheel moves the camera in and out."""
        self.distance *= ZOOM_PER_NOTCH ** scroll_y

    def on_key_press(self, symbol, modifiers) -> None:
        """Space pauses, R resets, Esc quits."""
        if symbol == key.SPACE:
            self.stepper.toggle_pause()
        elif symbol == key.R:
            self.stepper.reset()
        elif symbol == key.ESCAPE:
            self.close()

    def on_close(self) -> None:
        """Stop stepping, then close."""
        self.stepper.stop()
        super().on_close()

    def run(self, interval: float = 1 / 120) -> None:
        """Run the window loop until it closes."""
        pyglet.app.run(interval)

    def _fit(self, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            return
        self.centre = positions.mean(axis=0)
        reach = float(np.percentile(np.linalg.norm(positions - self.centre, axis=1), 99)) or 1.0
        self.distance = 1.1 * reach / np.sin(np.radians(FOV_DEGREES) / 2.0)
        spacing = 2.0 * reach / np.cbrt(len(positions))
        self.point_world = 0.5 * spacing

    def _count_frame(self) -> None:
        self._frames += 1
        now = time.perf_counter()
        if now - self._frames_since >= 1.0:
            self.frames_per_second = self._frames / (now - self._frames_since)
            self._frames, self._frames_since = 0, now
        if now - self._titled_at >= 0.25:
            self._titled_at = now
            if self.snapshot is not None:
                self.set_caption(title_line(self.snapshot, self.frames_per_second))
            elif self.stepper.error:
                self.set_caption(f"STOPPED: {self.stepper.error}")
