"""Cell mesh: the grid's cells on the GPU, see-through by their opacity, the far ones drawn first.

Contract: `CellMesh()` needs a current OpenGL 3.3 core context with RGB32F buffer
textures (a pyglet window; OpenGL 4.0 or later has them). `upload(corners, colours,
pattern)` takes what `grid_triangles` returns: float32 (n, k, 3) corner points of n
cells, uint8 (n, 4) RGBA per cell and the triangles of every cell as corner numbers
(at most MAX_PATTERN); n may change. It raises ValueError when the shapes disagree or
the cells do not fit a buffer texture. `draw(view_projection, width, height,
face_shade=(1, 1, 1), opacity_power=1)` draws them on the bound framebuffer, blended
over what is there: each cell in its colour, its opacity (alpha) raised to
`opacity_power`, each face darkened by `face_shade` for the axis it faces, hard edges,
no light of its own, no lines, both sides of every triangle. view_projection is a
(4, 4) row-major world-to-clip matrix (`orbit_camera`).

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

from physics.view.far_to_near import far_to_near

MAX_PATTERN = 36

_VERTEX = """#version 330 core
layout(location = 0) in int cell;
uniform samplerBuffer corners;
uniform samplerBuffer colours;
uniform int pattern[36];
uniform int corners_per_cell;
uniform mat4 view_projection;
out vec4 cell_colour;
out vec3 place;
void main() {
    place = texelFetch(corners, cell * corners_per_cell + pattern[gl_VertexID]).rgb;
    cell_colour = texelFetch(colours, cell);
    gl_Position = view_projection * vec4(place, 1.0);
}
"""

_FRAGMENT = """#version 330 core
in vec4 cell_colour;
in vec3 place;
uniform vec3 face_shade;
uniform float opacity_power;
out vec4 out_colour;
void main() {
    vec3 facing = cross(dFdx(place), dFdy(place));
    float size = dot(facing, facing);
    float shade = size > 0.0 ? dot(facing * facing, face_shade) / size : 1.0;
    out_colour = vec4(cell_colour.rgb * shade, pow(cell_colour.a, opacity_power));
}
"""


class CellMesh:
    """The cells' corners and colours in two buffer textures, their drawing order, and the program."""

    def __init__(self) -> None:
        self.program = _program(_VERTEX, _FRAGMENT)
        self.count, self.corners_per_cell, self.pattern_length = 0, 0, 0
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
        padded = np.zeros(MAX_PATTERN, dtype=np.int32)
        padded[:len(pattern)] = pattern
        gl.glUniform1iv(self._where("pattern"), MAX_PATTERN, padded.ctypes.data_as(ctypes.POINTER(gl.GLint)))
        gl.glUniform1i(self._where("corners_per_cell"), corners.shape[1])
        gl.glUseProgram(0)
        self.count, self.corners_per_cell, self.pattern_length = corners.shape[0], corners.shape[1], len(pattern)
        self._centres = corners.mean(axis=1) if self.count else np.zeros((0, 3), dtype=np.float32)
        self._sorted_for = None

    def draw(self, view_projection: np.ndarray, width: int, height: int,
             face_shade: tuple = (1.0, 1.0, 1.0), opacity_power: float = 1.0) -> None:
        """Draw the cells far to near, each blended over what lies behind it."""
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
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self.pattern_length, self.count)
        gl.glBindVertexArray(0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_BLEND)

    def _where(self, name: str) -> int:
        return gl.glGetUniformLocation(self.program, name.encode())


def _program(vertex: str, fragment: str) -> int:
    """Compile and link the two shaders (pyglet's own program class does not know buffer textures)."""
    program = gl.glCreateProgram()
    for kind, text in ((gl.GL_VERTEX_SHADER, vertex), (gl.GL_FRAGMENT_SHADER, fragment)):
        shader = gl.glCreateShader(kind)
        source = ctypes.cast(ctypes.c_char_p(text.encode()), ctypes.POINTER(gl.GLchar))
        gl.glShaderSource(shader, 1, ctypes.byref(source), None)
        gl.glCompileShader(shader)
        _raise_unless(shader, gl.GL_COMPILE_STATUS, gl.glGetShaderiv, gl.glGetShaderInfoLog)
        gl.glAttachShader(program, shader)
    gl.glLinkProgram(program)
    _raise_unless(program, gl.GL_LINK_STATUS, gl.glGetProgramiv, gl.glGetProgramInfoLog)
    return program


def _raise_unless(made: int, status: int, read_status, read_log) -> None:
    """Raise RuntimeError with the driver's log when a shader or program was not made."""
    passed = gl.GLint()
    read_status(made, status, ctypes.byref(passed))
    if not passed.value:
        log = ctypes.create_string_buffer(4096)
        read_log(made, 4096, None, log)
        raise RuntimeError(f"shader: {log.value.decode(errors='replace')}")
