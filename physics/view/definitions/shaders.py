"""Shaders: the two small programs that draw the swirl's streaks, on the card.

A streak is the way one tracer came, a line from mark to mark. It is red (plus fluid)
or blue (a joint's minus fluid) and nothing else; what changes is how solid it is: by
the energy moving there (the square of how far the fluid went in one mark) and by its
age, the newest end solid and the oldest clear. So the colour is the swirl itself:
solid where the fluid turns fast round the core, clear where it stands, and no box is
drawn anywhere. `FULL` and `THICK` only draw; they were set by eye once (2026-11-18).
"""
from physics.view.definitions.camera import FOCAL, NEAREST

FULL = 0.6     # fluid that goes this many boxes in one mark is drawn fully solid
THICK = 0.002  # a streak's thickness in cell edges; thinner than a picture dot, it is clearer

VERTEX = f"""#version 330 core
in vec4 place;  // a mark's place in boxes, and how far the tracer went to reach it
in float fade;  // 0 at the oldest mark of a tail, 1 at the newest
uniform vec3 eye, right, up, forward;
uniform float across, tall, aspect;
out float solid;
out float minus;
void main() {{
    vec3 spot = place.xyz / across - 0.5 - eye;
    float depth = dot(spot, forward);
    gl_Position = vec4({FOCAL} * dot(spot, right) / aspect, {FOCAL} * dot(spot, up),
                       depth - {NEAREST}, depth);
    minus = float(place.w < 0.0);  // a joint's fluid: its way gone is given below 0
    float energy = min(place.w * place.w / {FULL * FULL}, 1.0);
    float dots_thick = {THICK * FOCAL * 0.5} * tall / max(depth, {0.5 * NEAREST});
    solid = fade * energy * min(dots_thick, 1.0);
}}"""

FRAGMENT = """#version 330 core
in float solid;
in float minus;
out vec4 colour;
void main() { colour = vec4(mix(vec3(1.0, 0.15, 0.10), vec3(0.15, 0.35, 1.0), minus), solid); }"""
