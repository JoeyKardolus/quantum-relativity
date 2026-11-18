"""Whole picture: the whole grid in three dimensions as an image, drawn by the GPU off screen.

Contract: `WholePicture(width=WIDTH, height=HEIGHT, shown=SHOW_BOTH)` opens a hidden pyglet
window (it needs a display and the GPU, as the viewer does; call `use_gpu_driver()`
before the first import of this module). Calling it, `picture(pieces, low, high)`,
draws the whole grid exactly as the window does (`grid_triangles`, `CellMesh`: opacity
by packing, far cells first, faces shaded) over black and returns uint8 (height,
width, 3) RGB, row 0 at the top. The camera looks from the corner at lattice index 0
(CORNER) and is fitted to the box from `low` to `high` (`view_fit`): pass the same two
for every frame of a run and nothing jumps between frames.
"""
from __future__ import annotations

import numpy as np
import pyglet
from pyglet import gl

from physics.view.colours.definitions.palette import WHOLE_POWER
from physics.view.drawing.definitions.block_faces import FACE_SHADE
from physics.view.drawing.definitions.camera_marks import CORNER, FOV_DEGREES
from physics.view.drawing.definitions.orbit_camera import orbit_camera
from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH
from physics.view.drawing.logic.cell_mesh import CellMesh
from physics.view.drawing.logic.grid_triangles import grid_triangles
from physics.view.drawing.logic.view_fit import view_fit
from physics.view.layout.definitions.pieces import Pieces
from physics.view.pictures.definitions.picture_marks import HEIGHT, WIDTH


class WholePicture:
    """A hidden window, a mesh and an off-screen image to draw the whole grid into."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT, shown: int = SHOW_BOTH) -> None:
        self.width, self.height, self.shown = width, height, shown
        self.window = pyglet.window.Window(width=width, height=height, visible=False)
        self.window.switch_to()
        self.mesh = CellMesh()
        self.target = pyglet.image.Framebuffer()
        self.target.attach_texture(pyglet.image.Texture.create(width, height))

    def __call__(self, pieces: Pieces, low, high) -> np.ndarray:
        """An image of the whole grid, seen from the corner."""
        self.window.switch_to()
        corners, colours, pattern, _ = grid_triangles(pieces, self.shown, None)
        self.mesh.upload(corners, colours, pattern)
        centre, distance, _ = view_fit(low, high, FOV_DEGREES)
        camera = orbit_camera(centre, distance, CORNER[0], CORNER[1], self.width / self.height, FOV_DEGREES)
        self.target.bind()
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.mesh.draw(camera, self.width, self.height, FACE_SHADE, WHOLE_POWER)
        pixels = (gl.GLubyte * (self.width * self.height * 3))()
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels)
        self.target.unbind()
        return np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)[::-1].copy()
