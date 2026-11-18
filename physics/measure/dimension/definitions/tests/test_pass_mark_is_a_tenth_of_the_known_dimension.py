"""The pass mark by hand: within a tenth of the known dimension, on either side."""
import pytest

from physics.measure.dimension.definitions.pass_mark import WITHIN, passes


def test_the_mark_is_a_tenth():
    assert WITHIN == 0.1


@pytest.mark.parametrize("measured, known, verdict", [(2.71, 3, True), (3.29, 3, True), (2.69, 3, False),
                                                      (3.31, 3, False), (2.0, 2, True), (2.21, 2, False)])
def test_a_measure_passes_only_within_a_tenth_of_the_known_dimension(measured, known, verdict):
    assert passes(measured, known) is verdict


def test_a_known_dimension_of_zero_is_refused():
    with pytest.raises(ValueError):
        passes(1.0, 0)
