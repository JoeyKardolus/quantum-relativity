"""Cell window: the GPU window that draws a stepper's snapshots and takes the controls.

Contract: `CellWindow(stepper)` opens a resizable pyglet window (vsync off) that, on
every draw, uploads the stepper's newest snapshot (if any) to a `CellMesh` and draws it
through an `orbit_camera`, over black: the whole grid with its faces shaded and its
opacity on the whole grid's scale (FACE_SHADE, WHOLE_POWER), the slice as it is.
Positions are in grid lengths (the sum of cell lengths). On
the first snapshot, after each reset (a new generation), whenever the grid outgrows the
last fit (`view_fit`, REFIT_GROWTH) and on F, the view is fitted to the grid's bounds;
turning is kept. It starts on what the stepper starts on: the whole grid, seen from
the corner at lattice index 0.
Controls: space starts and pauses, C switches between the whole grid and the slice
through the middle (the slice turns the camera to face it, looking along z; the whole
grid turns it back to the corner), arrow keys or [ and ] move the slice one lattice layer,
1 shows only energy volumes, 2 only joints, 3 both, mouse drag rotates, wheel zooms,
F fits the view, R resets the source (and the view), Esc quits. The title
(`title_line`) refreshes 4 times a second. Closing the window stops the stepper.
`run(interval)` runs the pyglet loop, drawing every `interval` seconds.
"""
from __future__ import annotations

import time

import numpy as np
import pyglet
from pyglet.window import key

from physics.view.cell_colours import WHOLE_POWER
from physics.view.cell_mesh import CellMesh
from physics.view.grid_triangles import FACE_SHADE, SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS
from physics.view.orbit_camera import orbit_camera
from physics.view.title_line import title_line
from physics.view.view_fit import view_fit

FOV_DEGREES = 45.0
TURN_PER_PIXEL = 0.01
ZOOM_PER_NOTCH = 0.9
FACING = (0.0, 0.0)
CORNER = (0.6 + np.pi, -0.35)
_SHOWN_BY_KEY = {key._1: SHOW_ENERGY_VOLUMES, key._2: SHOW_JOINTS, key._3: SHOW_BOTH,
                 key.NUM_1: SHOW_ENERGY_VOLUMES, key.NUM_2: SHOW_JOINTS, key.NUM_3: SHOW_BOTH}
_MOVE_BY_KEY = {key.UP: 1, key.RIGHT: 1, key.BRACKETRIGHT: 1,
                key.DOWN: -1, key.LEFT: -1, key.BRACKETLEFT: -1}


class CellWindow(pyglet.window.Window):
    """The window: the grid's cells from the stepper, an orbit camera, keys and a status title."""

    def __init__(self, stepper, width: int = 1280, height: int = 800) -> None:
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

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        """Drag turns the view around the centre."""
        self.yaw -= dx * TURN_PER_PIXEL
        self.pitch = float(np.clip(self.pitch - dy * TURN_PER_PIXEL, -1.55, 1.55))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y) -> None:
        """Wheel moves the camera in and out."""
        self.distance *= ZOOM_PER_NOTCH ** scroll_y

    def on_key_press(self, symbol, modifiers) -> None:
        """Space, C, arrows, [ / ], 1 / 2 / 3, F, R and Esc, as the module contract says."""
        if symbol == key.SPACE:
            self.stepper.toggle_pause()
        elif symbol in _SHOWN_BY_KEY:
            self.stepper.show(_SHOWN_BY_KEY[symbol])
        elif symbol == key.C:
            self.stepper.toggle_whole()
            self.yaw, self.pitch = CORNER if self.stepper.whole else FACING
        elif symbol in _MOVE_BY_KEY:
            self.stepper.move_slice(_MOVE_BY_KEY[symbol])
        elif symbol == key.F:
            self.fitted_extent = None
            if self.snapshot is not None:
                self._fit(self.snapshot.low, self.snapshot.high)
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
        if now - self._titled_at >= 0.25:
            self._titled_at = now
            if self.snapshot is not None:
                self.set_caption(title_line(self.snapshot, self.frames_per_second))
            elif self.stepper.error:
                self.set_caption(f"STOPPED: {self.stepper.error}")
