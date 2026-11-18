"""Pair rotation: the exact step of two waves that trade fluid with each other.

Contract: `pair_rotation(amount_a, phase_a, amount_b, phase_b, turn)` (Warp func, all
float64) returns `(new_amount_a, new_phase_a, new_amount_b, new_phase_b)`. A wave is
`sqrt(amount) e^{i phase}`. What the two waves have in common, `(a + b) / 2`, stands as
it is; what they differ by, `(a - b) / 2`, turns by `2 turn`:
  a' = (a + b) / 2 + e^{2 i turn} (a - b) / 2
  b' = (a + b) / 2 - e^{2 i turn} (a - b) / 2
That keeps `amount_a + amount_b` (up to float64 rounding) and hands everything over at
a turn of pi / 2. Which way the fluid goes is set by the phases: it flows from the wave
whose phase is ahead. Two waves that are equal are left exactly as they are, so a fluid
of one phase stands still whatever the number of partners a box has: a box at the side
of its cell's block is not left behind, and the block's shape is no force.
The new phases are in (-pi, pi]; a wave that comes out exactly 0 keeps the phase it had.
Nothing is rounded to whole quanta here (whole_move.py does that). Used for two boxes
inside one cell and, in `cross` (axiom 4), for two boxes that face each other across a joint. It is how
waves add and cancel (axiom 7).
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})

_HALF = wp.constant(wp.float64(0.5))
_TWICE = wp.constant(wp.float64(2.0))


@wp.func
def pair_rotation(amount_a: wp.float64, phase_a: wp.float64, amount_b: wp.float64, phase_b: wp.float64,
                  turn: wp.float64):
    """The two waves' amounts and phases after their difference has turned by twice `turn`."""
    root_a = wp.sqrt(amount_a)
    root_b = wp.sqrt(amount_b)
    real_a = root_a * wp.cos(phase_a)
    imag_a = root_a * wp.sin(phase_a)
    real_b = root_b * wp.cos(phase_b)
    imag_b = root_b * wp.sin(phase_b)
    common_real = _HALF * (real_a + real_b)
    common_imag = _HALF * (imag_a + imag_b)
    apart_real = _HALF * (real_a - real_b)
    apart_imag = _HALF * (imag_a - imag_b)
    cosine = wp.cos(_TWICE * turn)
    sine = wp.sin(_TWICE * turn)
    turned_real = cosine * apart_real - sine * apart_imag
    turned_imag = sine * apart_real + cosine * apart_imag
    new_real_a = common_real + turned_real
    new_imag_a = common_imag + turned_imag
    new_real_b = common_real - turned_real
    new_imag_b = common_imag - turned_imag
    new_amount_a = new_real_a * new_real_a + new_imag_a * new_imag_a
    new_amount_b = new_real_b * new_real_b + new_imag_b * new_imag_b
    new_phase_a = phase_a
    if new_amount_a > wp.float64(0.0):
        new_phase_a = wp.atan2(new_imag_a, new_real_a)
    new_phase_b = phase_b
    if new_amount_b > wp.float64(0.0):
        new_phase_b = wp.atan2(new_imag_b, new_real_b)
    return new_amount_a, new_phase_a, new_amount_b, new_phase_b
