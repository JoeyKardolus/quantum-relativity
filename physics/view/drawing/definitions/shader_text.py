"""Shader text: the two GLSL programs the mesh draws the cells with.

Contract: constants only. `VERTEX` places one corner of one cell: the cell is an
instance, its corners and colour sit in two buffer textures, and `pattern` names the
corner each vertex of the triangle list is. `FRAGMENT` colours a point of a face: the
cell's colour, darkened by `face_shade` for the axis the face looks along, its opacity
raised to `opacity_power`. For the outlines the same two draw lines: `inset` above 0
moves every corner that share of the way to the cell's middle, and `outline` above 0
is the opacity every line is drawn with, in the cell's plain hue. The pattern array is
MAX_PATTERN long (block_faces.py).
"""
from __future__ import annotations

VERTEX = """#version 330 core
layout(location = 0) in int cell;
uniform samplerBuffer corners;
uniform samplerBuffer colours;
uniform int pattern[36];
uniform int corners_per_cell;
uniform mat4 view_projection;
uniform float inset;
out vec4 cell_colour;
out vec3 place;
void main() {
    place = texelFetch(corners, cell * corners_per_cell + pattern[gl_VertexID]).rgb;
    if (inset > 0.0) {
        vec3 middle = vec3(0.0);
        for (int corner = 0; corner < corners_per_cell; corner++)
            middle += texelFetch(corners, cell * corners_per_cell + corner).rgb;
        place = mix(place, middle / float(corners_per_cell), inset);
    }
    cell_colour = texelFetch(colours, cell);
    gl_Position = view_projection * vec4(place, 1.0);
}
"""

FRAGMENT = """#version 330 core
in vec4 cell_colour;
in vec3 place;
uniform vec3 face_shade;
uniform float opacity_power;
uniform float outline;
out vec4 out_colour;
void main() {
    if (outline > 0.0) {
        out_colour = vec4(cell_colour.rgb, outline);
        return;
    }
    vec3 facing = cross(dFdx(place), dFdy(place));
    float size = dot(facing, facing);
    float shade = size > 0.0 ? dot(facing * facing, face_shade) / size : 1.0;
    out_colour = vec4(cell_colour.rgb * shade, pow(cell_colour.a, opacity_power));
}
"""
