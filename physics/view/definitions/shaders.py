"""Shaders: the two small programs that draw the tracers' marks, on the card.

A streak is the way one tracer came, a line from mark to mark; a dot is one mark where
it stands, so fluid that stands still shows too. Both are red (plus fluid) or blue (a
joint's minus fluid) and nothing else; what changes is how solid: by what the mark
carries, and by its age, the newest end solid and the oldest clear. Many marks over each
other go from see-through to solid, so the colour is the fluid itself and no box is
drawn anywhere. `THICK`, `DOT` and `STILL` only draw; they were set by eye once.
"""
from physics.view.definitions.camera import FOCAL, NEAREST

THICK = 0.002  # a streak's thickness in cell edges; thinner than a picture dot, it is clearer.
# A grid's eye stands further off by its volumes along, and its streaks are that much thicker.
DOT = 0.07     # a dot's width in cell edges: wide enough that a cell's dots lie over each other
STILL = 0.3    # how solid a dot's middle is beside a streak

VERTEX = f"""#version 330 core
in vec4 place;  // a mark's place in boxes, and how solid it is (below 0: a joint's, blue)
in float fade;  // 0 at the oldest mark of a tail, 1 at the newest
uniform vec3 eye, right, up, forward;
uniform float across, tall, aspect, thick, dots;
out float solid;
out float minus;
void main() {{
    vec3 spot = place.xyz / across - 0.5 - eye;
    float depth = dot(spot, forward);
    gl_Position = vec4({FOCAL} * dot(spot, right) / aspect, {FOCAL} * dot(spot, up),
                       depth - {NEAREST}, depth);
    minus = float(place.w < 0.0);
    float edge = {FOCAL * 0.5} * tall / max(depth, {0.5 * NEAREST});  // a cell edge in picture dots
    gl_PointSize = max({DOT} * edge, 1.0);
    solid = fade * abs(place.w) * min(thick * {THICK} * edge, 1.0);
}}"""

FRAGMENT = f"""#version 330 core
in float solid;
in float minus;
uniform float dots;  // 1 while the dots are drawn: round, solid in the middle, clear at the rim
out vec4 colour;
void main() {{
    vec2 off = dots * (2.0 * gl_PointCoord - 1.0);
    float round = mix(1.0, {STILL} * max(1.0 - dot(off, off), 0.0), dots);
    colour = vec4(mix(vec3(1.0, 0.15, 0.10), vec3(0.15, 0.35, 1.0), minus), solid * round);
}}"""
