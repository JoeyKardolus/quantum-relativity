"""Point cloud: GL points of light, small fixed squares, summed in a float buffer, then shown on a fixed curve.

Contract: `PointCloud()` needs a current OpenGL 3.3 core context (a pyglet window).
`upload(positions, colours)` copies (n, 3) float32 positions and (n, 4) uint8
colours (RGB, and the point's light as a log byte, `light_code.py`) into two GPU
buffers (orphaned each time, GL_STREAM_DRAW); n may change. `draw(view_projection,
point_scale, width, height)` draws every point as a hard-edged square
SQUARE_PIXELS across, the same at every zoom (no round dot, no soft rim), adding
RGB x light into a half-float buffer the window's size (additive, depth test and
depth write off, so draw order never matters and millions of faint squares sum
without being rounded away as an 8-bit screen would). point_scale / (clip w) is
a box's true size in pixels; where it is smaller than the square, the square is
dimmed by (true size / SQUARE_PIXELS)^2, so zooming out does not brighten a cloud
by piling more squares on each pixel. Then the sum is shown on the
framebuffer that was bound, through 1 - exp(-sum) per channel: a fixed curve,
never scaled by the frame's own brightest pixel. view_projection is a (4, 4)
row-major world-to-clip matrix (`orbit_camera`). No link lines.
"""
from __future__ import annotations

import ctypes

import numpy as np
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram

from physics.view.light_code import LOG_DIMMEST, LOG_SPAN

SQUARE_PIXELS = 3.0

_VERTEX = f"""#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 colour;
uniform mat4 view_projection;
uniform float point_scale;
out vec3 dot_colour;
out float dot_light;
void main() {{
    vec4 clip = view_projection * vec4(position, 1.0);
    gl_Position = clip;
    float size = point_scale / max(clip.w, 1e-6);
    float held = {SQUARE_PIXELS!r};
    gl_PointSize = held;
    float code = floor(colour.a * 255.0 + 0.5);
    float light = code < 0.5 ? 0.0 : exp({LOG_DIMMEST!r} + {LOG_SPAN!r} * (code - 1.0) / 254.0);
    dot_light = light * min(1.0, (size / held) * (size / held));
    dot_colour = colour.rgb;
}}
"""

_FRAGMENT = """#version 330 core
in vec3 dot_colour;
in float dot_light;
out vec4 out_colour;
void main() {
    out_colour = vec4(dot_colour * dot_light, 1.0);
}
"""

_SCREEN_VERTEX = """#version 330 core
out vec2 place;
void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    place = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
"""

_SCREEN_FRAGMENT = """#version 330 core
in vec2 place;
uniform sampler2D summed;
out vec4 out_colour;
void main() {
    vec3 light = max(texture(summed, place).rgb, vec3(0.0));
    out_colour = vec4(vec3(1.0) - exp(-light), 1.0);
}
"""


class PointCloud:
    """Two GPU buffers of points, a float buffer they sum into, and the programs that draw them."""

    def __init__(self) -> None:
        self.program = ShaderProgram(Shader(_VERTEX, "vertex"), Shader(_FRAGMENT, "fragment"))
        self.screen = ShaderProgram(Shader(_SCREEN_VERTEX, "vertex"),
                                    Shader(_SCREEN_FRAGMENT, "fragment"))
        self.count = 0
        self._array = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self._array))
        self._screen_array = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self._screen_array))
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
        self._target = None

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

    def draw(self, view_projection: np.ndarray, point_scale: float, width: int, height: int) -> None:
        """Sum every point's light in the float buffer, then show it on the bound framebuffer."""
        shown_on = gl.GLint()
        gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING, ctypes.byref(shown_on))
        frame, texture = self._target_for(width, height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, frame)
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        self.program.use()
        self.program["view_projection"] = tuple(np.asarray(view_projection, dtype=float).T.ravel())
        self.program["point_scale"] = float(point_scale)
        gl.glBindVertexArray(self._array)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        self.program.stop()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, shown_on.value)
        gl.glDisable(gl.GL_BLEND)
        self.screen.use()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        self.screen["summed"] = 0
        gl.glBindVertexArray(self._screen_array)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        self.screen.stop()

    def _target_for(self, width: int, height: int) -> tuple[int, int]:
        """The half-float buffer of this size (made again when the window's size changes)."""
        if self._target is not None and self._target[0] == (width, height):
            return self._target[1], self._target[2]
        if self._target is not None:
            gl.glDeleteFramebuffers(1, ctypes.byref(gl.GLuint(self._target[1])))
            gl.glDeleteTextures(1, ctypes.byref(gl.GLuint(self._target[2])))
        texture = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(texture))
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, width, height, 0,
                        gl.GL_RGBA, gl.GL_HALF_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        frame = gl.GLuint()
        gl.glGenFramebuffers(1, ctypes.byref(frame))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, frame)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D,
                                  texture, 0)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"half-float light buffer {width}x{height} incomplete: {status:#x}")
        self._target = ((width, height), frame.value, texture.value)
        return frame.value, texture.value
