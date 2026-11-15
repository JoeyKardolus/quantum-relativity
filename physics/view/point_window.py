"""Point window: the GPU window that draws a stepper's snapshots and takes the controls.

Contract: `PointWindow(stepper)` opens a resizable pyglet window (vsync off) that, on
every draw, uploads the stepper's newest snapshot (if any) to a `PointCloud` and
draws it through an `orbit_camera`. Positions are box centres in box units. On the
first snapshot and after each reset (a new generation) the view is fitted to the
occupied boxes (the snapshot's low and high corners): centre at their middle,
distance so their widest extent fills about half the view (room for space to grow
into the empty margin); a box is POINT_BOXES across, which only sets how much a
fixed-size square is dimmed when boxes pile onto one pixel (`point_cloud.py`). Controls: mouse drag
rotates, wheel zooms, space starts and pauses, 1 shows only energy volumes,
2 only joints, 3 both, R resets the source (and the view), Esc quits. The title (`title_line`) refreshes 4 times a
second. Closing the window stops the stepper. `run(interval)` runs the pyglet loop,
drawing every `interval` seconds (0: as fast as it can).
"""
from __future__ import annotations

import time

import numpy as np
import pyglet
from pyglet.window import key

from physics.view.box_points import SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS
from physics.view.orbit_camera import orbit_camera
from physics.view.point_cloud import PointCloud
from physics.view.title_line import title_line

FOV_DEGREES = 45.0
TURN_PER_PIXEL = 0.01
ZOOM_PER_NOTCH = 0.9
POINT_BOXES = 1.0
_SHOWN_BY_KEY = {key._1: SHOW_ENERGY_VOLUMES, key._2: SHOW_JOINTS, key._3: SHOW_BOTH,
                 key.NUM_1: SHOW_ENERGY_VOLUMES, key.NUM_2: SHOW_JOINTS, key.NUM_3: SHOW_BOTH}


class PointWindow(pyglet.window.Window):
    """The window: points from the stepper, an orbit camera, keys and a status title."""

    def __init__(self, stepper, width: int = 1280, height: int = 800) -> None:
        super().__init__(width=width, height=height, resizable=True, vsync=False,
                         caption="quantum-relativity")
        self.stepper = stepper
        self.cloud = PointCloud()
        self.snapshot = None
        self.generation = -1
        self.centre, self.distance = np.zeros(3), 1.0
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
                self._fit(snapshot.low, snapshot.high)
                self.generation = snapshot.generation
            self.snapshot = snapshot
        width, height = self.get_framebuffer_size()
        pyglet.gl.glViewport(0, 0, width, height)
        pyglet.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.clear()
        camera = orbit_camera(self.centre, self.distance, self.yaw, self.pitch,
                              width / max(height, 1), FOV_DEGREES)
        focal = 1.0 / np.tan(np.radians(FOV_DEGREES) / 2.0)
        self.cloud.draw(camera, POINT_BOXES * focal * height / 2.0, width, height)
        self._count_frame()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        """Drag turns the view around the centre."""
        self.yaw -= dx * TURN_PER_PIXEL
        self.pitch = float(np.clip(self.pitch - dy * TURN_PER_PIXEL, -1.55, 1.55))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y) -> None:
        """Wheel moves the camera in and out."""
        self.distance *= ZOOM_PER_NOTCH ** scroll_y

    def on_key_press(self, symbol, modifiers) -> None:
        """Space starts and pauses, 1 / 2 / 3 show energy volumes / joints / both, R resets, Esc quits."""
        if symbol == key.SPACE:
            self.stepper.toggle_pause()
        elif symbol in _SHOWN_BY_KEY:
            self.stepper.show(_SHOWN_BY_KEY[symbol])
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

    def _fit(self, low: np.ndarray, high: np.ndarray) -> None:
        extent = max(float(np.max(high - low)), 1.0)
        self.centre = (np.asarray(low, float) + np.asarray(high, float)) / 2.0
        self.distance = 1.1 * extent / np.sin(np.radians(FOV_DEGREES) / 2.0)

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
