"""Link program: two GLSL texts compiled and linked into one OpenGL program.

Contract: `link_program(vertex, fragment)` needs a current OpenGL context and returns
the program's number. Raises RuntimeError with the driver's own log when a shader does
not compile or the program does not link. (pyglet's own program class does not know
buffer textures, so the mesh links its own.)
"""
from __future__ import annotations

import ctypes

from pyglet import gl


def link_program(vertex: str, fragment: str) -> int:
    """Compile the two shaders and link them."""
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
