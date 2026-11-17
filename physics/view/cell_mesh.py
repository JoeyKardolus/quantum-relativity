"""Cell mesh: GL triangles of the grid's cells, flat colours, the nearest in front.

Contract: `CellMesh()` needs a current OpenGL 3.3 core context (a pyglet window).
`upload(positions, colours)` copies (n, 3) float32 triangle corners and (n, 4) uint8
colours (RGB, alpha unused), three rows per triangle, into two GPU buffers (orphaned
each time, GL_STREAM_DRAW); n may change. `draw(view_projection, width, height)` draws
them on the bound framebuffer, depth-tested with no blending and both sides of every
triangle, each in its colour as given: hard edges, no light of its own, no lines.
view_projection is a (4, 4) row-major world-to-clip matrix (`orbit_camera`). The
framebuffer needs a depth buffer (pyglet's default config has one).
"""
from __future__ import annotations

import ctypes

import numpy as np
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

_VERTEX = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 colour;
uniform mat4 view_projection;
out vec3 cell_colour;
void main() {
    gl_Position = view_projection * vec4(position, 1.0);
    cell_colour = colour.rgb;
}
"""

_FRAGMENT = """#version 330 core
in vec3 cell_colour;
out vec4 out_colour;
void main() {
    out_colour = vec4(cell_colour, 1.0);
}
"""


class CellMesh:
    """Two GPU buffers of triangle corners and colours, and the program that draws them."""

    def __init__(self) -> None:
        self.program = ShaderProgram(Shader(_VERTEX, "vertex"), Shader(_FRAGMENT, "fragment"))
        self.count = 0
        self._array = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self._array))
        self._buffers = (gl.GLuint * 2)()
        gl.glGenBuffers(2, self._buffers)
        gl.glBindVertexArray(self._array)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._buffers[0])
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._buffers[1])
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_UNSIGNED_BYTE, gl.GL_TRUE, 0, None)
        gl.glBindVertexArray(0)

    def upload(self, positions: np.ndarray, colours: np.ndarray) -> None:
        """Copy triangle corners and colours to the GPU."""
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        colours = np.ascontiguousarray(colours, dtype=np.uint8)
        if positions.shape[0] != colours.shape[0] or positions.shape[0] % 3:
            raise ValueError(f"{positions.shape[0]} corners and {colours.shape[0]} colours are not whole triangles")
        for buffer, values in zip(self._buffers, (positions, colours)):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, values.nbytes, values.ctypes.data, gl.GL_STREAM_DRAW)
        self.count = positions.shape[0]

    def draw(self, view_projection: np.ndarray, width: int, height: int) -> None:
        """Draw the triangles, the nearest in front."""
        gl.glViewport(0, 0, width, height)
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDepthMask(gl.GL_TRUE)
        self.program.use()
        self.program["view_projection"] = tuple(np.asarray(view_projection, dtype=float).T.ravel())
        gl.glBindVertexArray(self._array)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.count)
        gl.glBindVertexArray(0)
        self.program.stop()
        gl.glDisable(gl.GL_DEPTH_TEST)
