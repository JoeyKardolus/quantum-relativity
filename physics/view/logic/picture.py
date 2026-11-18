"""Picture: one drawing of the swirl as bytes, made on the card without showing it.

The same painter that draws the window draws into a picture kept on the card, and
that one picture is read back. It is for saved pictures and tests, never for a frame.
"""
import numpy as np
import pyglet
from pyglet import gl

from physics.view.logic.swirl_painter import SwirlPainter


def picture(painter: SwirlPainter, across: int, yaw: float, pitch: float, distance: float,
            width: int, height: int) -> np.ndarray:
    """What the painter draws, as (height, width, 3) bytes, top row first."""
    frame = pyglet.image.Framebuffer()
    frame.attach_texture(pyglet.image.Texture.create(width, height))
    frame.bind()
    painter.draw(across, yaw, pitch, distance, width, height)
    pixels = (gl.GLubyte * (3 * width * height))()
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels)
    frame.unbind()
    frame.delete()
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)[::-1]
