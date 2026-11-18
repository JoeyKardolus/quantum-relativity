"""Two equal waves stand still, and the rotation keeps the two boxes' fluid together."""
import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import TURN, pair_rotation, wave_of


def test_two_equal_waves_stand_still():
    wave = wave_of(wp.int32(3000), wp.float32(0.7))
    after_a, after_b = pair_rotation(wave, wave, wp.float64(TURN))
    assert list(after_a) == list(wave) and list(after_b) == list(wave)


def test_the_rotation_keeps_the_fluid_of_the_two_boxes():
    wave_a, wave_b = wave_of(wp.int32(3000), wp.float32(0.7)), wave_of(wp.int32(1200), wp.float32(-2.1))
    after_a, after_b = pair_rotation(wave_a, wave_b, wp.float64(TURN))
    assert abs(np.dot(after_a, after_a) + np.dot(after_b, after_b) - 4200.0) < 1e-9
    assert np.dot(after_b, after_b) != 1200.0  # fluid did move: the phases differ
