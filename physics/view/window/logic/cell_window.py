"""Cell window: the GPU window that draws a stepper's snapshots and takes the keys.

Contract: `CellWindow(stepper, width=WIDTH, height=HEIGHT)` opens a resizable pyglet
window (vsync off) that, on every draw, uploads the stepper's newest snapshot (if any)
to a `CellMesh` and draws it through an `orbit_camera`, over black: the whole grid with
its faces shaded and its opacity on the whole grid's scale (FACE_SHADE, WHOLE_POWER),
the slice as it is. Positions are in grid lengths (the sum of cell lengths). It starts
on what the stepper starts on: the whole grid, seen from the corner at lattice index 0.
On the first snapshot, after each reset, whenever the grid outgrows the last fit
(`view_fit`) and on F, the view is fitted to the grid's bounds; turning is kept.
Keys are `KEY_ACTIONS`: C turns the camera to face the slice and back to the corner
for the whole grid. Mouse drag turns the view, the wheel zooms. The title
(`title_line`) refreshes every TITLE_EVERY seconds. Closing the window stops the
stepper. `run(interval)` runs the pyglet loop, drawing every `interval` seconds.
"""
from __future__ import annotations

import time

import numpy as np
import pyglet
from pyglet.window import key

from physics.view.colours.definitions.palette import WHOLE_POWER
from physics.view.drawing.definitions.block_faces import FACE_SHADE
from physics.view.drawing.definitions.camera_marks import CORNER, FACING, FOV_DEGREES
from physics.view.drawing.definitions.orbit_camera import orbit_camera
from physics.view.drawing.logic.cell_mesh import CellMesh
from physics.view.drawing.logic.view_fit import view_fit
from physics.view.window.definitions.key_actions import KEY_ACTIONS
from physics.view.window.definitions.window_marks import (HEIGHT, PITCH_LIMIT, TITLE_EVERY, TURN_PER_PIXEL, WIDTH,
                                                          ZOOM_PER_NOTCH)
from physics.view.window.logic.title_line import title_line

_ACTION_BY_SYMBOL = {getattr(key, name): action for name, action in KEY_ACTIONS.items()}


class CellWindow(pyglet.window.Window):
    """The window: the grid's cells from the stepper, an orbit camera, keys and a status title."""

    def __init__(self, stepper, width: int = WIDTH, height: int = HEIGHT) -> None:
        super().__init__(width=width, height=height, resizable=True, vsync=False,
                         caption="quantum-relativity")
        self.stepper = stepper
        self.mesh = CellMesh()
        self.snapshot = None
        self.generation = -1
        self.centre, self.distance = np.zeros(3), 1.0
        self.fitted_extent = None
        self.yaw, self.pitch = CORNER if stepper.whole else FACING
        self._frames, self._frames_since = 0, time.perf_counter()
        self.frames_per_second = 0.0
        self._titled_at = 0.0

    def on_draw(self) -> None:
        """Take the newest snapshot, draw it, count the frame."""
        snapshot = self.stepper.latest()
        if snapshot is not None:
            self.mesh.upload(snapshot.corners, snapshot.colours, snapshot.pattern)
            if snapshot.generation != self.generation:
                self.fitted_extent = None
                self.generation = snapshot.generation
            self._fit(snapshot.low, snapshot.high)
            self.snapshot = snapshot
        width, height = self.get_framebuffer_size()
        pyglet.gl.glViewport(0, 0, width, height)
        pyglet.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.clear()
        camera = orbit_camera(self.centre, self.distance, self.yaw, self.pitch,
                              width / max(height, 1), FOV_DEGREES)
        if self.snapshot is not None and self.snapshot.layer is None:
            self.mesh.draw(camera, width, height, FACE_SHADE, WHOLE_POWER)
        else:
            self.mesh.draw(camera, width, height)
        self._count_frame()

    def on_mouse_drag(self, pointer_x, pointer_y, dx, dy, buttons, modifiers) -> None:
        """Drag turns the view around the centre."""
        self.yaw -= dx * TURN_PER_PIXEL
        self.pitch = float(np.clip(self.pitch - dy * TURN_PER_PIXEL, -PITCH_LIMIT, PITCH_LIMIT))

    def on_mouse_scroll(self, pointer_x, pointer_y, scroll_x, scroll_y) -> None:
        """Wheel moves the camera in and out."""
        self.distance *= ZOOM_PER_NOTCH ** scroll_y

    def on_key_press(self, symbol, modifiers) -> None:
        """Do what `KEY_ACTIONS` says the key does; any other key does nothing."""
        action, argument = _ACTION_BY_SYMBOL.get(symbol, (None, None))
        if action == "pause":
            self.stepper.toggle_pause()
        elif action == "show":
            self.stepper.show(argument)
        elif action == "whole":
            self.stepper.toggle_whole()
            self.yaw, self.pitch = CORNER if self.stepper.whole else FACING
        elif action == "move":
            self.stepper.move_slice(argument)
        elif action == "fit":
            self.fitted_extent = None
            if self.snapshot is not None:
                self._fit(self.snapshot.low, self.snapshot.high)
        elif action == "reset":
            self.stepper.reset()
        elif action == "close":
            self.close()

    def on_close(self) -> None:
        """Stop stepping, then close."""
        self.stepper.stop()
        super().on_close()

    def run(self, interval: float = 1 / 120) -> None:
        """Run the window loop until it closes."""
        pyglet.app.run(interval)

    def _fit(self, low: np.ndarray, high: np.ndarray) -> None:
        """Fit the camera to the grid's bounds when one is due (`view_fit`)."""
        fit = view_fit(low, high, FOV_DEGREES, self.fitted_extent)
        if fit is not None:
            self.centre, self.distance, self.fitted_extent = fit

    def _count_frame(self) -> None:
        self._frames += 1
        now = time.perf_counter()
        if now - self._frames_since >= 1.0:
            self.frames_per_second = self._frames / (now - self._frames_since)
            self._frames, self._frames_since = 0, now
        if now - self._titled_at >= TITLE_EVERY:
            self._titled_at = now
            if self.snapshot is not None:
                self.set_caption(title_line(self.snapshot, self.frames_per_second))
            elif self.stepper.error:
                self.set_caption(f"STOPPED: {self.stepper.error}")
