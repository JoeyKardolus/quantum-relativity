"""Point cloud: plain GL points on the GPU, round, additive, no depth write.

Contract: `PointCloud()` needs a current OpenGL 3.3 core context (a pyglet window).
`upload(positions, colours)` copies (n, 3) float32 positions and (n, 4) uint8 RGBA
colours into two GPU buffers (orphaned each time, GL_STREAM_DRAW); n may change.
`draw(view_projection, point_scale)` draws every point as a round dot whose diameter
in pixels is point_scale / (clip w), clamped to [MIN_PIXELS, MAX_PIXELS]; the dot
fades at its rim; blending is additive (src alpha, one), depth test and depth write
off, so draw order never matters and overlapping faint dots add up. view_projection
is a (4, 4) row-major world-to-clip matrix (`orbit_camera`). No link lines.
"""
from __future__ import annotations

import ctypes

import numpy as np
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

MIN_PIXELS = 1.5
MAX_PIXELS = 32.0

_VERTEX = f"""#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 colour;
uniform mat4 view_projection;
uniform float point_scale;
out vec4 dot_colour;
void main() {{
    vec4 clip = view_projection * vec4(position, 1.0);
    gl_Position = clip;
    gl_PointSize = clamp(point_scale / max(clip.w, 1e-6), {MIN_PIXELS}, {MAX_PIXELS});
    dot_colour = colour;
}}
"""

_FRAGMENT = """#version 330 core
in vec4 dot_colour;
out vec4 out_colour;
void main() {
    vec2 offset = gl_PointCoord * 2.0 - 1.0;
    float reach = dot(offset, offset);
    if (reach > 1.0) discard;
    out_colour = vec4(dot_colour.rgb, dot_colour.a * (1.0 - smoothstep(0.5, 1.0, reach)));
}
"""


class PointCloud:
    """Two GPU buffers of points and the program that draws them as round dots."""

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
        """Copy positions and colours to the GPU."""
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        colours = np.ascontiguousarray(colours, dtype=np.uint8)
        if positions.shape[0] != colours.shape[0]:
            raise ValueError(f"{positions.shape[0]} positions but {colours.shape[0]} colours")
        for buffer, values in zip(self._buffers, (positions, colours)):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, values.nbytes, values.ctypes.data, gl.GL_STREAM_DRAW)
        self.count = positions.shape[0]

    def draw(self, view_projection: np.ndarray, point_scale: float) -> None:
        """Draw every point additively, as round dots sized by distance."""
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        self.program.use()
        self.program["view_projection"] = tuple(np.asarray(view_projection, dtype=float).T.ravel())
        self.program["point_scale"] = float(point_scale)
        gl.glBindVertexArray(self._array)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        gl.glBindVertexArray(0)
        self.program.stop()
