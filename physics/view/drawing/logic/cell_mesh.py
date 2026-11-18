"""Cell mesh: the grid's cells on the GPU, see-through by their opacity, the far ones drawn first.

Contract: `CellMesh()` needs a current OpenGL 3.3 core context with RGB32F buffer
textures (a pyglet window; OpenGL 4.0 or later has them). `upload(corners, colours,
pattern)` takes what `grid_triangles` returns: float32 (n, k, 3) corner points of n
cells, uint8 (n, 4) RGBA per cell and the triangles of every cell as corner numbers
(at most MAX_PATTERN); n may change. It raises ValueError when the shapes disagree or
the cells do not fit a buffer texture. `draw(view_projection, width, height,
face_shade=(1, 1, 1), opacity_power=1, outline=0)` draws them on the bound framebuffer,
blended over what is there: each cell in its colour, its opacity (alpha) raised to
`opacity_power`, each face darkened by `face_shade` for the axis it faces, hard edges,
no light of its own, both sides of every triangle. With `outline` above 0 every cell's
outline is then drawn over them in its plain hue at that opacity, whatever its fill
(FACE_LINES for four-cornered cells, BLOCK_LINES for blocks), set in by OUTLINE_INSET so
neighbours' outlines lie side by side. view_projection is a (4, 4) row-major
world-to-clip matrix (`orbit_camera`).

Depth is read by order, not by a depth buffer: the cells are sorted far to near by
their centres (`far_to_near`) whenever the cells or the view changed, and drawn in
that order, so a solid cell hides what is behind it and a thin one tints it. Only the
order (four bytes a cell) goes to the GPU per sort: corners and colours sit in two
buffer textures and every cell is one instance. The faces within one cell are not
sorted; they share a colour, so only their shade can come out a little off.
"""
from __future__ import annotations

import ctypes

import numpy as np
from pyglet import gl

from physics.view.colours.definitions.palette import OUTLINE_INSET
from physics.view.drawing.definitions.block_faces import BLOCK_LINES, FACE_LINES, MAX_PATTERN
from physics.view.drawing.definitions.shader_text import FRAGMENT, VERTEX
from physics.view.drawing.logic.far_to_near import far_to_near
from physics.view.drawing.logic.link_program import link_program


class CellMesh:
    """The cells' corners and colours in two buffer textures, their drawing order, and the program."""

    def __init__(self) -> None:
        self.program = link_program(VERTEX, FRAGMENT)
        self.count, self.corners_per_cell, self.pattern_length = 0, 0, 0
        self._pattern = np.zeros(0, dtype=np.int32)
        self._centres = np.zeros((0, 3), dtype=np.float32)
        self._sorted_for = None
        limit = gl.GLint()
        gl.glGetIntegerv(gl.GL_MAX_TEXTURE_BUFFER_SIZE, ctypes.byref(limit))
        self._most_texels = int(limit.value)
        self._array = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self._array))
        self._buffers = (gl.GLuint * 3)()
        gl.glGenBuffers(3, self._buffers)
        self._textures = (gl.GLuint * 2)()
        gl.glGenTextures(2, self._textures)
        for texture, buffer, texel in zip(self._textures, self._buffers, (gl.GL_RGB32F, gl.GL_RGBA8)):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
            gl.glBindTexture(gl.GL_TEXTURE_BUFFER, texture)
            gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, texel, buffer)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)
        gl.glBindVertexArray(self._array)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._buffers[2])
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribIPointer(0, 1, gl.GL_UNSIGNED_INT, 0, None)
        gl.glVertexAttribDivisor(0, 1)
        gl.glBindVertexArray(0)

    def upload(self, corners: np.ndarray, colours: np.ndarray, pattern: np.ndarray) -> None:
        """Copy the cells' corners, colours and triangle pattern to the GPU."""
        corners = np.ascontiguousarray(corners, dtype=np.float32)
        colours = np.ascontiguousarray(colours, dtype=np.uint8)
        pattern = np.ascontiguousarray(pattern, dtype=np.int32)
        if corners.ndim != 3 or corners.shape[2] != 3 or colours.shape != (corners.shape[0], 4):
            raise ValueError(f"corners {corners.shape} and colours {colours.shape} are not (n, k, 3) and (n, 4)")
        if len(pattern) % 3 or len(pattern) > MAX_PATTERN or (len(pattern) and pattern.max() >= corners.shape[1]):
            raise ValueError(f"pattern of {len(pattern)} corner numbers does not make triangles of {corners.shape[1]} corners")
        if corners.shape[0] * corners.shape[1] > self._most_texels:
            raise ValueError(f"{corners.shape[0]} cells of {corners.shape[1]} corners are more than the GPU's "
                             f"buffer texture holds ({self._most_texels} points)")
        for buffer, values in zip(self._buffers, (corners, colours)):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, values.nbytes, values.ctypes.data, gl.GL_STREAM_DRAW)
        gl.glUseProgram(self.program)
        gl.glUniform1i(self._where("corners_per_cell"), corners.shape[1])
        gl.glUseProgram(0)
        self._pattern = pattern
        self.count, self.corners_per_cell, self.pattern_length = corners.shape[0], corners.shape[1], len(pattern)
        self._centres = corners.mean(axis=1) if self.count else np.zeros((0, 3), dtype=np.float32)
        self._sorted_for = None

    def draw(self, view_projection: np.ndarray, width: int, height: int,
             face_shade: tuple = (1.0, 1.0, 1.0), opacity_power: float = 1.0, outline: float = 0.0) -> None:
        """Draw the cells far to near, each blended over what lies behind it, then their outlines."""
        matrix = np.asarray(view_projection, dtype=np.float32)
        if self._sorted_for is None or not np.array_equal(self._sorted_for, matrix):
            order = far_to_near(self._centres, matrix)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._buffers[2])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, order.nbytes, order.ctypes.data, gl.GL_STREAM_DRAW)
            self._sorted_for = matrix.copy()
        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self._where("view_projection"), 1, gl.GL_TRUE,
                              matrix.ctypes.data_as(ctypes.POINTER(gl.GLfloat)))
        gl.glUniform3f(self._where("face_shade"), *(float(shade) for shade in face_shade))
        gl.glUniform1f(self._where("opacity_power"), float(opacity_power))
        for unit, (name, texture) in enumerate(zip(("corners", "colours"), self._textures)):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_BUFFER, texture)
            gl.glUniform1i(self._where(name), unit)
        gl.glBindVertexArray(self._array)
        self._pass(self._pattern, 0.0, 0.0)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self.pattern_length, self.count)
        if outline > 0.0 and self.corners_per_cell in (4, 8):
            lines = FACE_LINES if self.corners_per_cell == 4 else BLOCK_LINES
            self._pass(lines, OUTLINE_INSET, outline)
            gl.glDrawArraysInstanced(gl.GL_LINES, 0, len(lines), self.count)
        gl.glBindVertexArray(0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_BLEND)

    def _pass(self, pattern: np.ndarray, inset: float, outline: float) -> None:
        """Set what one pass draws: the corner numbers of its vertices, the inset and the outline opacity."""
        padded = np.zeros(MAX_PATTERN, dtype=np.int32)
        padded[:len(pattern)] = pattern
        gl.glUniform1iv(self._where("pattern"), MAX_PATTERN, padded.ctypes.data_as(ctypes.POINTER(gl.GLint)))
        gl.glUniform1f(self._where("inset"), float(inset))
        gl.glUniform1f(self._where("outline"), float(outline))

    def _where(self, name: str) -> int:
        return gl.glGetUniformLocation(self.program, name.encode())
