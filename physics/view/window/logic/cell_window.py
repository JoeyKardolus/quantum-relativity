"""Cell window: the GPU window that draws a stepper's snapshots and takes the keys.

Contract: `CellWindow(stepper, width=WIDTH, height=HEIGHT)` opens a resizable pyglet
window (vsync off) that, on every draw, uploads the stepper's newest snapshot (if any)
and draws it through an `orbit_camera`, over black: the slice with its fills and every
cell's outline (`outline_opacity` of a place's width on screen, so the pattern always
shows and fades when seen from far), and, for the whole grid, the packed cells' blocks
over it, faces shaded, opacity on the whole grid's scale (FACE_SHADE, WHOLE_POWER).
Positions are in grid lengths. It starts on what the stepper starts on: the slice, seen
straight on. Every snapshot's frame is fitted (`view_fit`), so the grid stays still on
screen while the universe's size changes (the title carries the size); the wheel's zoom
is kept on top of the fit and F undoes it. Keys are
`KEY_ACTIONS`: C turns the camera to the corner at lattice index 0 for the whole grid
and back to face the slice. Mouse drag turns the view, the wheel zooms. The title
(`title_line`) refreshes every TITLE_EVERY seconds. Closing the window stops the
stepper. `run(interval)` runs the pyglet loop, drawing every `interval` seconds.
"""
from __future__ import annotations

import time

import numpy as np
import pyglet
from pyglet.window import key

from physics.view.colours.definitions.outline_opacity import outline_opacity
from physics.view.colours.definitions.palette import WHOLE_POWER
from physics.view.drawing.definitions.block_faces import FACE_SHADE
from physics.view.drawing.definitions.camera_marks import CORNER, FACE_FILL, FACING, FILL, FOV_DEGREES
from physics.view.drawing.definitions.orbit_camera import orbit_camera
from physics.view.drawing.definitions.place_pixels import place_pixels
from physics.view.drawing.logic.cell_mesh import CellMesh
from physics.view.drawing.logic.view_fit import view_fit
from physics.view.window.definitions.key_actions import KEY_ACTIONS
from physics.view.window.definitions.window_marks import (HEIGHT, PITCH_LIMIT, TITLE_EVERY, TURN_PER_PIXEL, WIDTH,
                                                          ZOOM_PER_NOTCH)
from physics.view.window.logic.title_line import title_line

_ACTION_BY_SYMBOL = {getattr(key, name): action for name, action in KEY_ACTIONS.items()}


class CellWindow(pyglet.window.Window):
    """The window: the slice and the blocks from the stepper, an orbit camera, keys and a status title."""

    def __init__(self, stepper, width: int = WIDTH, height: int = HEIGHT) -> None:
        super().__init__(width=width, height=height, resizable=True, vsync=False,
                         caption="quantum-relativity")
        self.stepper = stepper
        self.mesh = CellMesh()
        self.blocks = CellMesh()
        self.snapshot = None
        self.centre, self.distance, self.zoom = np.zeros(3), 1.0, 1.0
        self.yaw, self.pitch = CORNER if stepper.whole else FACING
        self._frames, self._frames_since = 0, time.perf_counter()
        self.frames_per_second = 0.0
        self._titled_at = 0.0

    def on_draw(self) -> None:
        """Take the newest snapshot, draw it, count the frame."""
        snapshot = self.stepper.latest()
        if snapshot is not None:
            self.mesh.upload(snapshot.corners, snapshot.colours, snapshot.pattern)
            if snapshot.blocks is not None:
                self.blocks.upload(*snapshot.blocks)
            self.snapshot = snapshot
            self._fit()
        width, height = self.get_framebuffer_size()
        pyglet.gl.glViewport(0, 0, width, height)
        pyglet.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.clear()
        camera = orbit_camera(self.centre, self.distance, self.yaw, self.pitch,
                              width / max(height, 1), FOV_DEGREES)
        if self.snapshot is not None:
            pixels = place_pixels(self.snapshot.low, self.snapshot.high, self.snapshot.places, self.distance, height)
            self.mesh.draw(camera, width, height, outline=outline_opacity(pixels))
            if self.snapshot.blocks is not None:
                self.blocks.draw(camera, width, height, FACE_SHADE, WHOLE_POWER)
        self._count_frame()

    def on_mouse_drag(self, pointer_x, pointer_y, dx, dy, buttons, modifiers) -> None:
        """Drag turns the view around the centre."""
        self.yaw -= dx * TURN_PER_PIXEL
        self.pitch = float(np.clip(self.pitch - dy * TURN_PER_PIXEL, -PITCH_LIMIT, PITCH_LIMIT))

    def on_mouse_scroll(self, pointer_x, pointer_y, scroll_x, scroll_y) -> None:
        """Wheel moves the camera in and out."""
        self.zoom *= ZOOM_PER_NOTCH ** scroll_y
        self._fit()

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
            self.zoom = 1.0
            self._fit()
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

    def _fit(self) -> None:
        """Fit the camera to the newest frame (`view_fit`), close for the slice and wider for the whole, times the zoom."""
        if self.snapshot is None:
            return
        self.centre, fitted, _ = view_fit(self.snapshot.low, self.snapshot.high, FOV_DEGREES,
                                          FACE_FILL if self.snapshot.blocks is None else FILL)
        self.distance = fitted * self.zoom

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
