"""The phase step by hand."""
import numpy as np
import pytest

from physics.measure.winding.definitions.phase_step import phase_step


def _at(phase):
    return np.array([np.exp(1j * phase)])


def test_a_small_step_is_the_plain_difference_whatever_the_amounts():
    assert phase_step(3.0 * _at(0.1), 0.5 * _at(0.4))[0] == pytest.approx(0.3)


def test_a_step_from_3_to_minus_3_radians_goes_forward_through_pi():
    assert phase_step(_at(3.0), _at(-3.0))[0] == pytest.approx(2.0 * np.pi - 6.0)
    assert phase_step(_at(-3.0), _at(3.0))[0] == pytest.approx(6.0 - 2.0 * np.pi)


@pytest.mark.parametrize("turns", [-2, -1, 0, 1, 3])
def test_the_steps_round_a_closed_ring_of_boxes_add_up_to_whole_turns(turns):
    ring = np.exp(1j * turns * 2.0 * np.pi * np.arange(16) / 16.0)
    assert phase_step(ring, np.roll(ring, -1)).sum() == pytest.approx(turns * 2.0 * np.pi)
