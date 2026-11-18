"""Swirl painter: draws the streaks as lines and, for a grid, every mark as a dot, on the card.

The lines' ends live in one buffer that OpenGL draws from. `load` copies new streaks
into it, the one small copy a tick costs (Warp and OpenGL share no memory under WSL2,
so it goes by the host); `draw` only sets the eye. The eye never goes into a kernel
and no picture is read back, so no view can reach outside an array.
"""
import numpy as np
import pyglet
import warp as wp
from pyglet import gl

from physics.view.definitions.camera import camera
from physics.view.definitions.shaders import FRAGMENT, VERTEX


class SwirlPainter:
    def __init__(self, marks: int, count: int):
        """Room on the card for `count` tails of `marks` marks; needs an open window."""
        shader = pyglet.graphics.shader
        self.thick = 1.0  # times `THICK`: a grid's volumes along, so it is as solid as one cell
        self.dots = False  # a grid's marks are dots too: fluid that stands still is seen
        self.program = shader.ShaderProgram(shader.Shader(VERTEX, "vertex"),
                                            shader.Shader(FRAGMENT, "fragment"))
        older = np.arange((marks - 1) * count)  # each line: a mark and the next newer one
        ends = np.stack([older, older + count], axis=1).ravel().tolist()
        fade = np.repeat(np.linspace(0.0, 1.0, marks, dtype=np.float32), count)
        self.lines = self.program.vertex_list_indexed(marks * count, gl.GL_LINES, ends,
                                                      fade=("f", fade))
        self.places = self.lines.domain.attrib_name_buffers["place"]
        self.ends = wp.array(ptr=self.places.data_ptr, dtype=wp.vec4, shape=(marks, count),
                             device="cpu")  # pyglet's own copy of the buffer, seen as an array

    def load(self, drawn: wp.array) -> None:
        """Hand new streaks to OpenGL: (marks, tracers) of four floats, as `streaks` gives them."""
        wp.copy(self.ends, drawn)
        self.places.invalidate_region(0, self.lines.count)

    def draw(self, across: int, yaw: float, pitch: float, distance: float,
             width: int, height: int) -> None:
        """Draw the streaks into a picture `width` by `height`, black behind them."""
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.program.use()
        for name, value in zip(("eye", "right", "up", "forward"), camera(yaw, pitch, distance)):
            self.program[name] = tuple(float(part) for part in value)
        self.program["across"], self.program["tall"] = float(across), float(height)
        self.program["aspect"], self.program["thick"] = width / max(height, 1), self.thick
        self.program["dots"] = 0.0
        self.lines.draw(gl.GL_LINES)
        if self.dots:
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            self.program["dots"] = 1.0
            gl.glDrawArrays(gl.GL_POINTS, self.lines.start, self.lines.count)
