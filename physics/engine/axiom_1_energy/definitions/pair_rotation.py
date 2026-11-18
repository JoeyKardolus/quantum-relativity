"""Pair rotation: how the fluid of two neighbouring boxes moves in one step.

A box's fluid is a wave, `sqrt(quanta) * e^(i phase)`, kept as (real, imaginary). What
two waves have in common stands as it is; what they differ by turns by twice `TURN`.
So two equal waves stand still, and fluid flows from the wave whose phase is ahead.
Layer 1 holds plus quanta only: how a joint's minus fluid moves is not decided yet.
"""
import warp as wp

# A chosen step size. No axiom sets it yet (the owner decides what does); it was never
# fitted to a picture, and the standing swirl and its turn per step follow from it.
TURN = 0.25


@wp.func
def wave_of(quanta: wp.int32, phase: wp.float32) -> wp.vec2d:
    """A box's wave from its whole quanta (0 or more) and its phase."""
    root = wp.sqrt(wp.float64(quanta))
    return wp.vec2d(root * wp.cos(wp.float64(phase)), root * wp.sin(wp.float64(phase)))


@wp.func
def phase_of(wave: wp.vec2d, old_phase: wp.float32) -> wp.float32:
    """A wave's phase in (-pi, pi]; a wave that is 0 keeps the phase it had."""
    if wp.dot(wave, wave) > wp.float64(0.0):
        return wp.float32(wp.atan2(wave[1], wave[0]))
    return old_phase


@wp.func
def pair_rotation(wave_a: wp.vec2d, wave_b: wp.vec2d, turn: wp.float64):
    """The two waves after their difference has turned by twice `turn`. Nothing is rounded."""
    common = (wave_a + wave_b) * wp.float64(0.5)
    apart = (wave_a - wave_b) * wp.float64(0.5)
    cosine = wp.cos(wp.float64(2.0) * turn)
    sine = wp.sin(wp.float64(2.0) * turn)
    turned = wp.vec2d(cosine * apart[0] - sine * apart[1], sine * apart[0] + cosine * apart[1])
    return common + turned, common - turned
